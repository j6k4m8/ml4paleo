import { describe, expect, it } from "vitest";
import { ApiError } from "#lib/api.ts";
import type { DeltaIn } from "./deltas";
import { MAX_DELTAS, type OpOut, OpQueue, type Outcome } from "./opqueue.svelte";

const delta = (x: number): DeltaIn => ({ key: [0, 0, x], base_version: 0, box: [0, 0, 0, 1, 1, 1], mask: "", value: 2, only_if: "any" });

function server() {
	const calls: { path: string; body: { client_op_id: string } }[] = [];
	let seq = 0;
	const failures: unknown[] = [];
	const send = async (path: string, body: unknown): Promise<OpOut> => {
		calls.push({ path, body: body as { client_op_id: string } });
		const failure = failures.shift();
		if (failure) throw failure;
		return { seq: ++seq, chunks: [] };
	};
	return { calls, failures, send };
}

async function settle(queue: OpQueue, timeout = 3000) {
	const start = Date.now();
	while (queue.pending > 0 && Date.now() - start < timeout) await new Promise((r) => setTimeout(r, 5));
}

describe("OpQueue", () => {
	it("sends edits in order, and undoes and redoes them by seq", async () => {
		const { calls, send } = server();
		const queue = new OpQueue("p", null, send);
		queue.edit([delta(0)]);
		queue.edit([delta(1)]);
		await settle(queue);
		queue.undo();
		queue.redo();
		await settle(queue);
		expect(calls.map((c) => c.path)).toEqual([
			"/api/projects/p/labels/ops",
			"/api/projects/p/labels/ops",
			"/api/projects/p/labels/ops/2/undo",
			"/api/projects/p/labels/ops/2/redo",
		]);
		expect(new Set(calls.map((c) => c.body.client_op_id)).size).toBe(4);
	});

	it("retries network failures with the same client_op_id", async () => {
		const { calls, failures, send } = server();
		failures.push(new TypeError("Failed to fetch"));
		const queue = new OpQueue("p", null, send);
		queue.edit([delta(0)]);
		await new Promise((r) => setTimeout(r, 20));
		expect(queue.offline).toBe(true);
		await settle(queue);
		expect(calls).toHaveLength(2);
		expect(calls[0]?.body.client_op_id).toBe(calls[1]?.body.client_op_id);
		expect(queue.offline).toBe(false);
	});

	it("reports strict conflicts and moves on", async () => {
		const { failures, send } = server();
		failures.push(new ApiError(409, { message: "Some chunks changed; reload them.", chunks: [[0, 0, 1]] }));
		const queue = new OpQueue("p", null, send);
		const outcomes: Outcome[] = [];
		queue.onOutcome((o) => outcomes.push(o));
		queue.edit([delta(1)], { strict: true });
		queue.edit([delta(2)]);
		await settle(queue);
		expect(outcomes[0]).toMatchObject({ conflict: [[0, 0, 1]] });
		expect("result" in outcomes[1]!).toBe(true);
	});

	it("splits edits too big for one op, and undoes them together", async () => {
		const { calls, send } = server();
		const queue = new OpQueue("p", null, send);
		const ops = queue.edit(Array.from({ length: MAX_DELTAS + 3 }, (_, i) => delta(i)));
		expect(ops).toHaveLength(2);
		await settle(queue);
		queue.undo();
		await settle(queue);
		expect(calls.map((c) => c.path.split("/labels/")[1])).toEqual(["ops", "ops", "ops/2/undo", "ops/1/undo"]);
		expect(queue.undoable).toBe(0);
		expect(queue.redoable).toBe(1);
	});

	it("skips undoing an edit that never saved", async () => {
		const { calls, failures, send } = server();
		failures.push(new ApiError(422, "label values [9] are not classes here"));
		const queue = new OpQueue("p", null, send);
		queue.edit([delta(0)]);
		await settle(queue);
		queue.undo();
		await settle(queue);
		expect(calls).toHaveLength(1);
		expect(queue.error).toContain("not classes");
	});

	it("drops an edit undone before it was sent, and sends it anew on redo", async () => {
		const { calls, failures, send } = server();
		failures.push(new TypeError("Failed to fetch"));
		const queue = new OpQueue("p", null, send);
		const outcomes: Outcome[] = [];
		const requeued: string[] = [];
		queue.onOutcome((o) => outcomes.push(o));
		queue.onRequeue((ops) => requeued.push(...ops.map((op) => op.local)));
		queue.edit([delta(0)]);
		const [second] = queue.edit([delta(1)]);
		queue.undo();
		expect(outcomes).toEqual([{ op: second, cancelled: true }]);
		queue.redo();
		expect(requeued).toHaveLength(1);
		expect(requeued[0]).not.toBe(second?.local);
		await settle(queue);
		const sent = calls.map((c) => c.body.client_op_id);
		expect(sent).not.toContain(second?.clientOpId);
		expect(calls.filter((c) => c.path.endsWith("/ops"))).toHaveLength(3);
	});

	it("saves undos of sent edits, with their seq", async () => {
		const { send } = server();
		const saved: { kind: string; seq?: number }[] = [];
		const storage = { load: async () => [], save: async (op: { kind: string; seq?: number }) => void saved.push(op), remove: async () => {} };
		const queue = new OpQueue("p", storage, send);
		queue.edit([delta(0)]);
		await settle(queue);
		queue.undo();
		expect(saved.map((op) => [op.kind, op.seq])).toEqual([
			["edit", undefined],
			["undo", 1],
		]);
	});

	it("lets the page adjust an edit just before it goes", async () => {
		const bodies: unknown[] = [];
		const queue = new OpQueue("p", null, async (_path, body) => {
			bodies.push(body);
			return { seq: 1, chunks: [] };
		});
		queue.beforeSend = (op) => ({ ...op, deltas: op.deltas.map((d) => ({ ...d, base_version: 7 })) });
		queue.edit([delta(0)], { strict: true });
		await settle(queue);
		expect((bodies[0] as { deltas: { base_version: number }[] }).deltas[0]?.base_version).toBe(7);
	});

	it("resumes edits a previous page left unsent", async () => {
		const { calls, send } = server();
		const saved = { kind: "edit" as const, local: "a", clientOpId: "c-1", deltas: [delta(0)], strict: false, tool: {} };
		const removed: string[] = [];
		const storage = { load: async () => [saved], save: async () => {}, remove: async (id: string) => void removed.push(id) };
		const queue = new OpQueue("p", storage, send);
		await queue.start();
		await settle(queue);
		expect(calls[0]?.body.client_op_id).toBe("c-1");
		await new Promise((r) => setTimeout(r, 5));
		expect(removed).toEqual(["a"]);
	});
});
