import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Busy, type Chunk, ChunkStore, type Loader, PATIENCE_MS, RETRY_GAP_MS } from "./chunks";

function chunk(bytes: number): Chunk {
	return { data: new Uint8Array(bytes), shape: [1, 1, bytes] };
}

/** A loader whose loads finish when the test says so. */
function controlled() {
	const calls: { id: string; signal: AbortSignal; at: number; finish: (bytes?: number) => void; fail: (e: unknown) => void }[] = [];
	const load = (id: string, signal: AbortSignal) =>
		new Promise<Chunk>((resolve, reject) => {
			calls.push({ id, signal, at: Date.now(), finish: (bytes = 10) => resolve(chunk(bytes)), fail: reject });
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

	it("says which load of a chunk is under way, and what a refresh was asked for", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		expect(store.loading("a")).toBe(first);
		expect(store.refreshing("a")).toBeUndefined();
		calls[0]?.finish();
		await first;
		expect(store.loading("a")).toBeUndefined();
		const again = store.refresh("a", 7);
		expect(store.refreshing("a")).toBe(7);
		calls[1]?.finish();
		await again;
		expect(store.refreshing("a")).toBeUndefined();
	});

	it("cancels many loads in a few passes over its lists, not one per load", () => {
		const { load } = controlled();
		const store = new ChunkStore(load, 1e9, 1);
		for (let i = 0; i < 500; i++) store.request(`c${i}`).catch(() => {});
		const filter = vi.spyOn(Array.prototype, "filter");
		try {
			store.keepOnly(new Set(["c0"]));
			expect(filter.mock.calls.length).toBeLessThanOrEqual(3);
		} finally {
			filter.mockRestore();
		}
	});

	it("cancels many loads at once, keeping the rest in order", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1e9, 1);
		const ids = Array.from({ length: 2000 }, (_, i) => `c${i}`);
		for (const id of ids) store.request(id).catch(() => {});
		store.keepOnly(new Set(["c0", "c1500", "c1999"]));
		calls[0]?.finish();
		await tick();
		calls[1]?.finish();
		await tick();
		expect(calls.map((c) => c.id)).toEqual(["c0", "c1500", "c1999"]);
	});

	it("keeps the cached copy when a refresh replaces another", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		calls[0]?.finish(10);
		const old = await first;
		store.refresh("a").catch(() => {});
		const again = store.refresh("a");
		expect(calls[1]?.signal.aborted).toBe(true);
		expect(store.peek("a")).toBe(old);
		calls[2]?.finish(30);
		expect((await again).data.byteLength).toBe(30);
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

describe("ChunkStore, for chunks the server is still making", () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	/** Lets the loads that settled run what follows (a store reacts a few promises on). */
	const settle = () => vi.advanceTimersByTimeAsync(0);
	/** What a store does with a load: how often each id was asked for, and when, in order, since `start`. */
	const asked = (calls: { id: string; at: number }[], start: number) => calls.map((c) => `${c.id}@${c.at - start}`);

	it("gives up the load's place at once, so the chunks behind it go on, and asks again after the delay", async () => {
		const { calls, load } = controlled();
		const start = Date.now();
		const store = new ChunkStore(load, 1000, 1);
		const a = store.request("a");
		store.request("b");
		store.request("c");
		calls[0]?.fail(new Busy(1000));
		await settle();
		expect(asked(calls, start)).toEqual(["a@0", "b@0"]);
		calls[1]?.finish();
		await settle();
		calls[2]?.finish();
		await settle();
		await vi.advanceTimersByTimeAsync(999);
		expect(calls).toHaveLength(3);
		await vi.advanceTimersByTimeAsync(1);
		expect(asked(calls, start)).toEqual(["a@0", "b@0", "c@0", "a@1000"]);
		calls[3]?.finish(20);
		expect((await a).data.byteLength).toBe(20);
		expect(store.isLoading("a")).toBe(false);
		expect(store.get("a")).toBeDefined();
		// Nothing is left to wake the store.
		expect(vi.getTimerCount()).toBe(0);
	});

	it("keeps the chunk loading while it waits, for the views asking and for a refresh's copy", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const first = store.request("a");
		calls[0]?.finish(10);
		const old = await first;
		const again = store.refresh("a", 7);
		calls[1]?.fail(new Busy(1000));
		await settle();
		expect(store.isLoading("a")).toBe(true);
		expect(store.loading("a")).toBe(again);
		expect(store.refreshing("a")).toBe(7);
		// The copy shown stays, and a view's asking for the chunk doesn't start another load.
		expect(store.get("a")).toBe(old);
		expect(await store.request("a")).toBe(old);
		expect(calls).toHaveLength(2);
		await vi.advanceTimersByTimeAsync(1000);
		expect(calls).toHaveLength(3);
		calls[2]?.finish(30);
		expect((await again).data.byteLength).toBe(30);
		expect(store.get("a")).not.toBe(old);
		expect(store.refreshing("a")).toBeUndefined();
	});

	it("loads the chunks never asked for first, whatever they rank against one asked again", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		store.want("xy", ["a", "b", "c"]);
		for (const id of ["a", "b", "c"]) store.request(id).catch(() => {});
		calls[0]?.fail(new Busy(100));
		await settle();
		// b holds the only place until after a is due.
		await vi.advanceTimersByTimeAsync(500);
		calls[1]?.finish();
		await settle();
		calls[2]?.finish();
		await settle();
		expect(calls.map((c) => c.id)).toEqual(["a", "b", "c", "a"]);
	});

	it("doesn't let chunks the server is making hold up the ones it isn't, however many wait", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1e9, 4);
		const coarse = Array.from({ length: 20 }, (_, i) => `coarse${i}`);
		const fine = Array.from({ length: 66 }, (_, i) => `fine${i}`);
		store.want("xy", [...coarse, ...fine]);
		for (const id of [...coarse, ...fine]) store.request(id).catch(() => {});
		// The server is making the coarse chunks, which takes a while, and sends the rest as they come.
		let answered = 0;
		while (answered < calls.length) {
			const call = calls[answered++]!;
			if (call.id.startsWith("coarse")) call.fail(new Busy(1000));
			else call.finish();
			await settle();
		}
		// None of the 66 waited for a coarse chunk to be made, or for a place a coarse chunk's waiting held.
		for (const id of fine) expect(store.peek(id)).toBeDefined();
		expect(calls).toHaveLength(86);
		for (const id of coarse) expect(store.isLoading(id)).toBe(true);
	});

	it("asks again for no more than one chunk every RETRY_GAP_MS, in the order they came due, however many wait", async () => {
		const { calls, load } = controlled();
		const start = Date.now();
		const store = new ChunkStore(load, 1e9, 8);
		const ids = ["a", "b", "c", "d", "e", "f"];
		for (const id of ids) store.request(id).catch(() => {});
		for (const call of [...calls]) call.fail(new Busy(1000));
		await settle();
		expect(calls).toHaveLength(6);
		await vi.advanceTimersByTimeAsync(999);
		expect(calls).toHaveLength(6);
		await vi.advanceTimersByTimeAsync(1);
		expect(asked(calls.slice(6), start)).toEqual(["a@1000"]);
		await vi.advanceTimersByTimeAsync(RETRY_GAP_MS - 1);
		expect(calls).toHaveLength(7);
		await vi.advanceTimersByTimeAsync(1);
		expect(asked(calls.slice(6), start)).toEqual(["a@1000", `b@${1000 + RETRY_GAP_MS}`]);
		await vi.advanceTimersByTimeAsync(10 * RETRY_GAP_MS);
		expect(asked(calls.slice(6), start)).toEqual(ids.map((id, i) => `${id}@${1000 + i * RETRY_GAP_MS}`));
	});

	it("takes turns among the chunks it asks again for, the longest waiting first", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1e9, 8);
		const ids = ["a", "b", "c", "d", "e", "f", "g", "h"];
		store.want("xy", ids);
		for (const id of ids) store.request(id).catch(() => {});
		// Every chunk is asked for again and answered busy again, over and over.
		let answered = 0;
		const busy = async () => {
			while (answered < calls.length) calls[answered++]?.fail(new Busy(1000));
			await settle();
		};
		await busy();
		for (let round = 0; round < 40; round++) {
			await vi.advanceTimersByTimeAsync(RETRY_GAP_MS);
			await busy();
		}
		// The best ranked mustn't keep the others waiting: each is asked for about as often.
		const times = ids.map((id) => calls.filter((c) => c.id === id).length);
		expect(Math.min(...times)).toBeGreaterThanOrEqual(4);
		expect(Math.max(...times) - Math.min(...times)).toBeLessThanOrEqual(1);
	});

	it("wakes for whichever chunk is due first, even if a later one parked earlier", async () => {
		const { calls, load } = controlled();
		const start = Date.now();
		const store = new ChunkStore(load, 1e9, 4);
		store.request("a").catch(() => {});
		store.request("b").catch(() => {});
		calls[0]?.fail(new Busy(5000));
		await settle();
		calls[1]?.fail(new Busy(1000));
		await settle();
		await vi.advanceTimersByTimeAsync(5000);
		expect(asked(calls, start)).toEqual(["a@0", "b@0", "b@1000", "a@5000"]);
	});

	it("asks again as often as the server says it is busy, each time after the delay it gives", async () => {
		const { calls, load } = controlled();
		const start = Date.now();
		const store = new ChunkStore(load, 1000, 1);
		const a = store.request("a");
		for (const delay of [1000, 3000]) {
			calls.at(-1)?.fail(new Busy(delay));
			await vi.advanceTimersByTimeAsync(delay);
		}
		calls.at(-1)?.finish();
		await a;
		expect(asked(calls, start)).toEqual(["a@0", "a@1000", "a@4000"]);
	});

	it("stops waiting for a chunk no view wants any more", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		store.want("xy", ["a", "b"]);
		const a = store.request("a");
		const b = store.request("b");
		const lost = a.catch((e: unknown) => e);
		b.catch(() => {});
		calls[0]?.fail(new Busy(1000));
		await settle();
		store.want("xy", ["b"]);
		const error = await lost;
		expect(error).toBeInstanceOf(DOMException);
		expect((error as DOMException).name).toBe("AbortError");
		expect(store.isLoading("a")).toBe(false);
		await vi.advanceTimersByTimeAsync(10_000);
		expect(calls.map((c) => c.id)).toEqual(["a", "b"]);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("stops waiting for every chunk when all are cancelled", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const waiting = ["a", "b", "c"].map((id) => store.request(id).catch((e: unknown) => e));
		for (const call of [...calls]) call.fail(new Busy(1000));
		await settle();
		store.keepOnly(new Set());
		for (const error of await Promise.all(waiting)) expect((error as DOMException).name).toBe("AbortError");
		await vi.advanceTimersByTimeAsync(10_000);
		expect(calls).toHaveLength(3);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("lets a refresh, or an invalidation, take the place of a load that waits", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const waiting = store.request("a");
		const lost = waiting.catch((e: unknown) => e);
		calls[0]?.fail(new Busy(1000));
		await settle();
		const fresh = store.refresh("a", 3);
		expect(((await lost) as DOMException).message).toBe("Changed while loading");
		// The refresh starts at once: it hasn't been asked for before.
		expect(calls.map((c) => c.id)).toEqual(["a", "a"]);
		expect(store.refreshing("a")).toBe(3);
		calls[1]?.fail(new Busy(1000));
		await settle();
		fresh.catch(() => {});
		store.invalidate("a");
		await expect(fresh).rejects.toThrow("Changed while loading");
		await vi.advanceTimersByTimeAsync(10_000);
		expect(calls).toHaveLength(2);
	});

	it("gives up once another wait would end past its patience, with the last answer", async () => {
		const { calls, load } = controlled();
		const start = Date.now();
		const store = new ChunkStore(load, 1000, 1);
		const a = store.request("a");
		const lost = a.catch((e: unknown) => e);
		const answers = [new Busy(50_000, "first"), new Busy(50_000, "second"), new Busy(50_000, "third")];
		for (const answer of answers.slice(0, 2)) {
			calls.at(-1)?.fail(answer);
			await vi.advanceTimersByTimeAsync(50_000);
		}
		expect(asked(calls, start)).toEqual(["a@0", "a@50000", "a@100000"]);
		calls[2]?.fail(answers[2]);
		const error = await lost;
		// A wait to 150 s would end past the 120 s the chunk is waited for.
		expect(error).toBe(answers[2]);
		expect((error as Busy).cause).toBe("third");
		expect(store.isLoading("a")).toBe(false);
		await vi.advanceTimersByTimeAsync(PATIENCE_MS);
		expect(calls).toHaveLength(3);
		expect(vi.getTimerCount()).toBe(0);
		// Asked for afterwards, the chunk gets its patience afresh.
		store.request("a").catch(() => {});
		expect(calls).toHaveLength(4);
		calls[3]?.fail(new Busy(50_000));
		await vi.advanceTimersByTimeAsync(50_000);
		expect(calls).toHaveLength(5);
	});

	it("is patient for two minutes by default", async () => {
		const load = vi.fn(async () => {
			throw new Busy(2000);
		});
		const store = new ChunkStore(load, 1000);
		const lost = store.request("a").catch((e: unknown) => e);
		await vi.advanceTimersByTimeAsync(PATIENCE_MS + 10_000);
		expect(await lost).toBeInstanceOf(Busy);
		// Asked for every two seconds, from 0 to 120 s.
		expect(load).toHaveBeenCalledTimes(61);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("counts the time the answers took against its patience, but not the time spent in line", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000, 1);
		const slow = store.request("slow").catch((e: unknown) => e);
		const queued = store.request("queued");
		// The first load takes 100 s to answer that the server is busy for another 30: a wait to 130.
		await vi.advanceTimersByTimeAsync(100_000);
		calls[0]?.fail(new Busy(30_000));
		expect(await slow).toBeInstanceOf(Busy);
		// The second spent that time in line, which isn't the server's doing.
		await settle();
		expect(calls.map((c) => c.id)).toEqual(["slow", "queued"]);
		calls[1]?.fail(new Busy(30_000));
		await vi.advanceTimersByTimeAsync(30_000);
		expect(calls.map((c) => c.id)).toEqual(["slow", "queued", "queued"]);
		calls[2]?.finish();
		await queued;
	});

	it("passes any other failure on at once, as it did", async () => {
		const { calls, load } = controlled();
		const store = new ChunkStore(load, 1000);
		const lost = store.request("a").catch((e: unknown) => e);
		const error = new Error("broken");
		calls[0]?.fail(error);
		expect(await lost).toBe(error);
		expect(store.isLoading("a")).toBe(false);
		await vi.advanceTimersByTimeAsync(10_000);
		expect(calls).toHaveLength(1);
	});
});

describe("ChunkStore, for chunks the server takes long to make", () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	const settle = () => vi.advanceTimersByTimeAsync(0);
	// Slow chunks' ids start with "s", the others' with "f"; four places, two for the slow.
	const store = (load: Loader) => new ChunkStore(load, 1e9, 4, { slow: (id) => id.startsWith("s"), places: 2 });
	const ask = (s: ChunkStore, ids: string[]) => {
		s.want("xy", ids);
		for (const id of ids) s.request(id).catch(() => {});
	};
	const started = (calls: { id: string }[]) => calls.map((c) => c.id);

	it("keeps places for the chunks the server only reads, however the chunks rank", async () => {
		const { calls, load } = controlled();
		const chunks = store(load);
		ask(chunks, ["s1", "s2", "s3", "s4", "f1", "f2", "f3", "f4"]);
		expect(started(calls)).toEqual(["s1", "s2", "f1", "f2"]);
		calls[0]?.finish();
		await settle();
		// A slow place is free again, and the next slow chunk takes it.
		expect(started(calls)).toEqual(["s1", "s2", "f1", "f2", "s3"]);
		calls[2]?.finish();
		await settle();
		// Both slow places are taken, so a place that frees goes to a chunk that isn't.
		expect(started(calls)).toEqual(["s1", "s2", "f1", "f2", "s3", "f3"]);
		calls[1]?.finish();
		calls[4]?.finish();
		await settle();
		expect(started(calls).slice(6)).toEqual(["s4", "f4"]);
	});

	it("keeps those places for them even when none waits now", async () => {
		const { calls, load } = controlled();
		const chunks = store(load);
		ask(chunks, ["s1", "s2", "s3", "s4"]);
		expect(started(calls)).toEqual(["s1", "s2"]);
		calls[1]?.finish();
		await settle();
		expect(started(calls)).toEqual(["s1", "s2", "s3"]);
		// A chunk that arrives meanwhile finds its places free.
		chunks.request("f1").catch(() => {});
		chunks.request("f2").catch(() => {});
		expect(started(calls)).toEqual(["s1", "s2", "s3", "f1", "f2"]);
	});

	it("gives a slow place up at once when the server answers busy, to the next slow chunk", async () => {
		const { calls, load } = controlled();
		const chunks = store(load);
		ask(chunks, ["s1", "s2", "s3", "s4"]);
		calls[0]?.fail(new Busy(1000));
		calls[1]?.fail(new Busy(1000));
		await settle();
		expect(started(calls)).toEqual(["s1", "s2", "s3", "s4"]);
	});

	it("asks again for a slow chunk only when a slow place is free", async () => {
		const { calls, load } = controlled();
		const chunks = store(load);
		ask(chunks, ["s1", "s2", "s3"]);
		calls[0]?.fail(new Busy(1000));
		await settle();
		// s3 took the place s1 left; s2 and s3 hold both.
		expect(started(calls)).toEqual(["s1", "s2", "s3"]);
		await vi.advanceTimersByTimeAsync(5000);
		expect(started(calls)).toEqual(["s1", "s2", "s3"]);
		calls[2]?.finish();
		await settle();
		expect(started(calls)).toEqual(["s1", "s2", "s3", "s1"]);
	});

	it("doesn't count a load it cancelled against the places once it has ended", async () => {
		const { calls, load } = controlled();
		const chunks = store(load);
		ask(chunks, ["s1", "s2", "s3"]);
		chunks.want("xy", ["s3"]);
		for (const call of calls.slice(0, 2)) call.fail(new DOMException("No longer needed", "AbortError"));
		await settle();
		expect(started(calls)).toEqual(["s1", "s2", "s3"]);
	});

	it("puts no limit on a store without one", async () => {
		const { calls, load } = controlled();
		const unlimited = new ChunkStore(load, 1e9, 4);
		ask(unlimited, ["s1", "s2", "s3", "s4"]);
		expect(started(calls)).toEqual(["s1", "s2", "s3", "s4"]);
	});
});
