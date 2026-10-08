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

	it("starts the most wanted queued load next", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		store.want("xy", ["a", "b", "c", "d"]);
		for (const id of ["d", "c", "b", "a"]) store.request(id).catch(() => {});
		calls[0]?.finish();
		await tick();
		calls[1]?.finish();
		await tick();
		calls[2]?.finish();
		await tick();
		expect(calls.map((c) => c.id)).toEqual(["d", "a", "b", "c"]);
	});

	it("lets views sharing the store take turns", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		store.want("xy", ["x1", "x2", "x3"]);
		store.want("xz", ["z1", "z2", "z3"]);
		for (const id of ["x1", "x2", "x3", "z1", "z2", "z3"]) store.request(id).catch(() => {});
		for (let i = 0; i < 5; i++) {
			calls[i]?.finish();
			await tick();
		}
		expect(calls.map((c) => c.id)).toEqual(["x1", "z1", "x2", "z2", "x3", "z3"]);
	});

	it("follows the view's latest order, cancelling what it left", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		store.want("xy", ["a", "b", "c"]);
		for (const id of ["a", "b", "c"]) store.request(id).catch(() => {});
		store.want("xy", ["c", "b"]);
		expect(calls[0]?.signal.aborted).toBe(true);
		calls[0]?.fail(new DOMException("Aborted", "AbortError"));
		await tick();
		calls[1]?.finish();
		await tick();
		expect(calls.map((c) => c.id)).toEqual(["a", "c", "b"]);
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

	it("keeps loads that any owner still wants", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		store.want("xy", new Set(["a"]));
		const a = store.request("a");
		store.want("xz", new Set(["b"]));
		const b = store.request("b");
		store.want("xy", new Set());
		await expect(a).rejects.toThrow("No longer needed");
		expect(calls[1]?.signal.aborted).toBe(false);
		calls[1]?.finish();
		await expect(b).resolves.toBeDefined();
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

	it("cancels a load that an invalidation overtakes", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const stale = store.request("a");
		store.invalidate("a");
		await expect(stale).rejects.toThrow("Changed while loading");
		expect(calls[0]?.signal.aborted).toBe(true);
		calls[0]?.finish();
		const fresh = store.request("a");
		expect(calls).toHaveLength(2);
		calls[1]?.finish(20);
		expect((await fresh).data.byteLength).toBe(20);
	});

	it("keeps chunks a view wants even over budget", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 15);
		store.want("xy", new Set(["a", "b"]));
		for (const id of ["a", "b"]) {
			const pending = store.request(id);
			calls.at(-1)?.finish();
			await pending;
		}
		expect(store.get("a")).toBeDefined();
		expect(store.get("b")).toBeDefined();
		store.want("xy", new Set(["b"]));
		const c = store.request("c");
		calls.at(-1)?.finish();
		await c;
		expect(store.get("a")).toBeUndefined();
	});

	it("evicts the chunks a view wants least first", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 25);
		for (const id of ["a", "b"]) {
			const pending = store.request(id);
			calls.at(-1)?.finish();
			await pending;
		}
		// Wanted but not kept, most wanted first.
		store.want("xy", ["b", "a"], []);
		const c = store.request("c");
		calls.at(-1)?.finish();
		await c;
		expect(store.peek("a")).toBeUndefined();
		expect(store.peek("b")).toBeDefined();
	});

	it("peeks without counting as a use", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 25);
		for (const id of ["a", "b"]) {
			const pending = store.request(id);
			calls.at(-1)?.finish();
			await pending;
		}
		expect(store.peek("a")).toBeDefined();
		expect(store.isLoading("c")).toBe(false);
		const c = store.request("c");
		expect(store.isLoading("c")).toBe(true);
		calls.at(-1)?.finish();
		await c;
		expect(store.isLoading("c")).toBe(false);
		expect(store.peek("a")).toBeUndefined();
	});

	it("forgets owners that want nothing", () => {
		const { load } = controlled();
		const store = new ChunkStore(load, 1000);
		store.want("accept:a", ["a"]);
		expect(store.isWanted("a")).toBe(true);
		store.want("accept:a", []);
		expect(store.isWanted("a")).toBe(false);
	});

	it("refreshes a chunk while the old copy stays in use", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		calls[0]?.finish(10);
		const old = await first;
		const seen: string[] = [];
		store.onLoad = (id) => seen.push(id);
		const fresh = store.refresh("a");
		expect(store.get("a")).toBe(old);
		calls[1]?.finish(30);
		expect((await fresh).data.byteLength).toBe(30);
		expect(store.get("a")).not.toBe(old);
		expect(store.bytes).toBe(30);
		expect(seen).toEqual(["a"]);
	});

	it("drops an out-of-date copy when its refresh is cancelled", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		calls[0]?.finish(10);
		await first;
		const refreshing = store.refresh("a");
		refreshing.catch(() => {});
		store.want("xy", new Set(["b"]));
		await expect(refreshing).rejects.toThrow("No longer needed");
		expect(store.get("a")).toBeUndefined();
	});

	it("survives a loader that throws", async () => {
		let n = 0;
		const store = new ChunkStore(
			(id) => {
				n++;
				if (id === "bad") throw new Error("broken");
				return Promise.resolve(chunk(1));
			},
			1000,
			1,
		);
		await expect(store.request("bad")).rejects.toThrow("broken");
		await expect(store.request("good")).resolves.toBeDefined();
		expect(n).toBe(2);
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
