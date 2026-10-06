import { describe, expect, it } from "vitest";
import { type Chunk, ChunkStore } from "./chunks";

function chunk(bytes: number): Chunk {
	return { data: new Uint8Array(bytes), shape: [1, 1, bytes] };
}

/** A loader whose loads finish when the test says so. */
function controlled() {
	const calls: { id: string; signal: AbortSignal; finish: (bytes?: number) => void; fail: (e: unknown) => void }[] = [];
	const load = (id: string, signal: AbortSignal) =>
		new Promise<Chunk>((resolve, reject) => {
			calls.push({ id, signal, finish: (bytes = 10) => resolve(chunk(bytes)), fail: reject });
		});
	return { calls, load };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("ChunkStore", () => {
	it("loads each chunk once however often it is asked for", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const a = store.request("a");
		const b = store.request("a");
		expect(calls).toHaveLength(1);
		calls[0]?.finish();
		expect(await a).toBe(await b);
		expect(await store.request("a")).toBe(await a);
		expect(calls).toHaveLength(1);
	});

	it("runs a limited number of loads at a time, in order", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 2);
		for (const id of ["a", "b", "c", "d"]) store.request(id).catch(() => {});
		expect(calls.map((c) => c.id)).toEqual(["a", "b"]);
		calls[1]?.finish();
		await tick();
		expect(calls.map((c) => c.id)).toEqual(["a", "b", "c"]);
	});

	it("cancels loads the view no longer needs", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		const a = store.request("a");
		const b = store.request("b");
		store.request("c").catch(() => {});
		store.keepOnly(new Set(["c"]));
		await expect(a).rejects.toThrow("No longer needed");
		await expect(b).rejects.toThrow("No longer needed");
		expect(calls[0]?.signal.aborted).toBe(true);
		// A cancelled load that finishes anyway isn't cached.
		calls[0]?.finish();
		await tick();
		expect(store.get("a")).toBeUndefined();
		expect(calls.map((c) => c.id)).toEqual(["a", "c"]);
	});

	it("can ask again for a chunk after cancelling it", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		store.request("a").catch(() => {});
		store.keepOnly(new Set());
		const again = store.request("a");
		calls[1]?.finish();
		await expect(again).resolves.toBeDefined();
	});

	it("drops the least recently used chunks over its budget, but not pinned ones", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 25);
		for (const id of ["a", "b"]) {
			const pending = store.request(id);
			calls.at(-1)?.finish();
			await pending;
		}
		store.pin("a");
		store.get("b");
		const c = store.request("c");
		calls.at(-1)?.finish();
		await c;
		expect(store.get("a")).toBeDefined();
		expect(store.get("b")).toBeUndefined();
		expect(store.bytes).toBe(20);
	});

	it("forgets failed loads so they can be retried", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		calls[0]?.fail(new Error("offline"));
		await expect(first).rejects.toThrow("offline");
		const second = store.request("a");
		expect(calls).toHaveLength(2);
		calls[1]?.finish();
		await expect(second).resolves.toBeDefined();
	});

	it("forgets invalidated chunks", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const a = store.request("a");
		calls[0]?.finish();
		await a;
		store.invalidate("a");
		expect(store.get("a")).toBeUndefined();
		expect(store.bytes).toBe(0);
	});
});
