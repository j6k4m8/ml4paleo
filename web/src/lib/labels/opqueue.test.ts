import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "#lib/api.ts";
import type { DeltaIn } from "./deltas";
import { batches, MAX_DELTAS, MAX_OP_BYTES, type OpOut, OpQueue, type Outcome, type Queued, saveState } from "./opqueue.svelte";

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

	it("says so while the server can't save, and carries on when it can", async () => {
		const { calls, failures, send } = server();
		failures.push(new ApiError(500, "Internal Server Error"));
		const queue = new OpQueue("p", null, send);
		queue.edit([delta(0)]);
		await new Promise((r) => setTimeout(r, 20));
		// Waiting to try again, and not "offline": the server answered.
		expect(queue.retrying).toBe(true);
		expect(queue.offline).toBe(false);
		expect(queue.pending).toBe(1);
		await settle(queue);
		expect(calls).toHaveLength(2);
		expect(calls[0]?.body.client_op_id).toBe(calls[1]?.body.client_op_id);
		expect(queue.retrying).toBe(false);
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

	it("keeps ops under the request size limit", () => {
		const big = (x: number): DeltaIn => ({ ...delta(x), mask: "a".repeat(MAX_OP_BYTES / 3) });
		const parts = batches([big(0), big(1), big(2), big(3)]);
		expect(parts.map((p) => p.length)).toEqual([2, 2]);
	});

	it("sends accepted predictions to be checked, and undoes them as one", async () => {
		const { calls, send } = server();
		const queue = new OpQueue("p", null, send);
		queue.editMany([[delta(0)], [delta(1)]], { accept: { prediction: "pred", roi: "roi" } });
		await settle(queue);
		expect(calls.map((c) => c.path)).toEqual(["/api/projects/p/labels/accept", "/api/projects/p/labels/accept"]);
		expect(calls[0]?.body).toMatchObject({ prediction_artifact_id: "pred", roi_id: "roi" });
		queue.undo();
		await settle(queue);
		expect(calls.slice(2).map((c) => c.path.split("/labels/")[1])).toEqual(["ops/2/undo", "ops/1/undo"]);
		expect(queue.undoable).toBe(0);
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

	it("sends the edits a previous page left unsent even if an edit went out before they were read", async () => {
		const sent: string[] = [];
		let answer!: () => void;
		const send = async (_path: string, body: unknown): Promise<OpOut> => {
			sent.push((body as { client_op_id: string }).client_op_id);
			if (sent.length === 1) await new Promise<void>((resolve) => (answer = resolve));
			return { seq: sent.length, chunks: [] };
		};
		const saved: Queued = { kind: "edit", local: "saved", clientOpId: "saved-op", deltas: [delta(1)], strict: false, tool: {} };
		let read!: (ops: Queued[]) => void;
		const removed: string[] = [];
		const storage = {
			load: () => new Promise<Queued[]>((resolve) => (read = resolve)),
			save: async () => {},
			remove: async (id: string) => void removed.push(id),
		};
		const queue = new OpQueue("p", storage, send);
		const outcomes: Outcome[] = [];
		queue.onOutcome((outcome) => outcomes.push(outcome));
		const starting = queue.start();
		const [mine] = queue.edit([delta(0)]);
		// The new edit goes out while the saved ones are still being read.
		await new Promise((r) => setTimeout(r, 5));
		read([saved]);
		await starting;
		answer();
		await settle(queue);
		await new Promise((r) => setTimeout(r, 5));
		// Each is sent once, and answered once: the new edit's answer doesn't take the saved edit's place.
		expect(sent).toEqual([mine!.clientOpId, "saved-op"]);
		expect(outcomes.map((o) => o.op.local)).toEqual([mine!.local, "saved"]);
		expect(removed.sort()).toEqual(["saved", mine!.local].sort());
		expect(queue.pending).toBe(0);
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

describe("the save status", () => {
	afterEach(() => vi.useRealTimers());

	/** A queue whose saves fail with `failures`, one for each try, and then work. */
	function failing(...failures: unknown[]) {
		const { failures: list, send } = server();
		list.push(...failures);
		vi.useFakeTimers();
		return new OpQueue("p", null, send);
	}
	const step = (ms: number) => vi.advanceTimersByTimeAsync(ms);

	it("follows the latest try, and is saved when one works", async () => {
		// The tries come 1 s, 2 s, and 4 s apart.
		const queue = failing(new ApiError(500, "x"), new TypeError("Failed to fetch"), new ApiError(503, "x"));
		queue.edit([delta(0)]);
		await step(1);
		expect(saveState(queue)).toBe("retrying");
		await step(1000);
		expect(saveState(queue)).toBe("offline");
		await step(2000);
		expect(saveState(queue)).toBe("retrying");
		await step(4000);
		expect(saveState(queue)).toBe("saved");
		queue.stop();
	});

	it("is saved when undoing the one edit that was waiting leaves nothing to save", async () => {
		for (const failure of [new ApiError(500, "x"), new TypeError("Failed to fetch")]) {
			const queue = failing(failure, failure, failure);
			queue.edit([delta(0)]);
			await step(1);
			expect(saveState(queue)).not.toBe("saved");
			queue.undo();
			expect(saveState(queue)).toBe("saved");
			await step(5000);
			expect(saveState(queue)).toBe("saved");
			queue.stop();
		}
	});

	it("keeps saying so while another edit still waits", async () => {
		const queue = failing(new ApiError(500, "x"), new ApiError(500, "x"));
		queue.edit([delta(0)]);
		queue.edit([delta(1)]);
		await step(1);
		queue.undo();
		expect(queue.pending).toBe(1);
		expect(saveState(queue)).toBe("retrying");
		queue.stop();
	});

	it("asks to sign in again whatever failed before, until a save works", async () => {
		const queue = failing(new ApiError(500, "x"), new ApiError(401, "no"));
		queue.edit([delta(0)]);
		await step(1);
		expect(saveState(queue)).toBe("retrying");
		await step(1000);
		expect(saveState(queue)).toBe("error");
		expect(queue.error).toMatch(/Sign in/);
		await step(2000);
		expect(saveState(queue)).toBe("saved");
		queue.stop();
	});

	it("shows a newer failure over the error an earlier edit left", async () => {
		const queue = failing(new ApiError(422, "label values [9] are not classes here"), new ApiError(500, "x"));
		queue.edit([delta(0)]);
		queue.edit([delta(1)]);
		await step(1);
		expect(queue.error).toContain("not classes");
		expect(saveState(queue)).toBe("retrying");
		queue.stop();
	});
});
