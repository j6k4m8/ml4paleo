import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "#lib/api.ts";
import { base64, packBits, zstdFrame } from "../labels/deltas";
import type { Chunk } from "./chunks";
import { LabelLayer, strictOn } from "./labels";
import type { WorkerPool } from "./loader";

/**
 * A worker pool whose label chunks come from a map the test controls. With
 * `held`, loads answer from the map as it is when they start but arrive
 * only when the test calls `release`. The next `failures.left` loads fail.
 */
function fakePool(server: Map<string, { value: number; version?: number }>, held = false) {
	const loads: string[] = [];
	const waiting: (() => void)[] = [];
	const failures = { left: 0 };
	const pool = {
		load: async (request: { region: [number, number][] }): Promise<Chunk> => {
			const id = request.region.map(([start]) => start / 64).join("/");
			loads.push(id);
			const { value, version } = server.get(id) ?? { value: 0, version: 0 };
			if (held) await new Promise<void>((resolve) => waiting.push(resolve));
			if (failures.left > 0) {
				failures.left -= 1;
				throw new Error("offline");
			}
			return { data: new Uint8Array(8).fill(value), shape: [2, 2, 2], version };
		},
	};
	const release = () => {
		for (const resolve of waiting.splice(0)) resolve();
	};
	return { pool: pool as unknown as WorkerPool, loads, release, failures };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
// For tests with fake timers: lets the promises already resolved run their callbacks.
const settled = async () => {
	for (let i = 0; i < 20; i++) await Promise.resolve();
};

vi.mock("#lib/api.ts", () => ({ api: vi.fn() }));

const delta = (value: number, key: [number, number, number] = [0, 0, 0]) => ({
	key,
	base_version: 0,
	box: [0, 0, 0, 2, 2, 2] as [number, number, number, number, number, number],
	mask: base64(zstdFrame(packBits(new Uint8Array(8).fill(1)))),
	value,
	only_if: "any",
});

/** The first voxel of a cached label chunk. */
function first(layer: LabelLayer, id: string): number | undefined {
	return (layer.store.get(id)?.data as Uint8Array | undefined)?.[0];
}

async function loaded(layer: LabelLayer, id: string) {
	layer.store.want("view", new Set([id]));
	return layer.store.request(id);
}

/** What views are told, as the layer tells them. */
function told(layer: LabelLayer): string[][] {
	const heard: string[][] = [];
	layer.onChange((ids) => heard.push(ids));
	return heard;
}

describe("LabelLayer", () => {
	it("shows an edit at once and keeps it across reloads until it settles", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		expect(first(layer, "0/0/0")).toBe(5);
		// Someone else's change arrives first; our edit stays on top.
		server.set("0/0/0", { value: 2, version: 4 });
		layer.reload(["0/0/0"]);
		await new Promise((r) => setTimeout(r, 0));
		expect(first(layer, "0/0/0")).toBe(5);
		expect(layer.store.get("0/0/0")?.version).toBe(4);
	});

	it("takes the version an edit made when the copy shown was the one before", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		await tick();
		expect(layer.versionOf("0/0/0")).toBe(4);
		expect(first(layer, "0/0/0")).toBe(5);
		expect(loads).toHaveLength(1);
		// Its own change event, arriving after, has nothing new to load.
		layer.changed([{ key: [0, 0, 0], version: 4 }]);
		await tick();
		expect(loads).toHaveLength(1);
	});

	it("keeps an edit showing on a copy others changed too, until a fresh copy comes", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		// Someone's edit, then ours, landed on the server.
		server.set("0/0/0", { value: 5, version: 7 });
		layer.settle("op", [{ key: [0, 0, 0], version: 7 }]);
		expect(first(layer, "0/0/0")).toBe(5);
		await tick();
		expect(loads).toHaveLength(2);
		expect(layer.versionOf("0/0/0")).toBe(7);
		expect(first(layer, "0/0/0")).toBe(5);
	});

	it("puts an edit back on a copy read just before it, which makes the copy that version", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		// Drawn while the chunk loads; the server applies it before the load lands.
		layer.applyLocal("op", [delta(5)]);
		server.set("0/0/0", { value: 5, version: 4 });
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		release();
		await loading;
		// The copy predates the edit, but with it put back is what the server has.
		expect(first(layer, "0/0/0")).toBe(5);
		expect(layer.versionOf("0/0/0")).toBe(4);
		await tick();
		expect(loads).toHaveLength(1);
		// Its change event has nothing new to load either.
		layer.changed([{ key: [0, 0, 0], version: 4 }]);
		expect(loads).toHaveLength(1);
	});

	it("puts an edit back on a copy others changed the chunk since, and loads the chunk once more", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 2 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		// Someone else's edit (3), then ours (4), reach the server before the load lands.
		server.set("0/0/0", { value: 5, version: 4 });
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		release();
		await loading;
		// The copy is from before both: it shows our edit, and stands for the
		// version before the other, so a strict edit on it is refused.
		expect(first(layer, "0/0/0")).toBe(5);
		expect(layer.versionOf("0/0/0")).toBe(2);
		await tick();
		expect(loads).toHaveLength(2);
		// The reload is for our edit's version: its change event doesn't start it over.
		layer.changed([{ key: [0, 0, 0], version: 4 }]);
		expect(loads).toHaveLength(2);
		release();
		await tick();
		expect(layer.versionOf("0/0/0")).toBe(4);
		expect(first(layer, "0/0/0")).toBe(5);
	});

	it("loads again the chunks an undo changed", async () => {
		const server = new Map([["0/0/0", { value: 5, version: 4 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		server.set("0/0/0", { value: 0, version: 5 });
		layer.changed([{ key: [0, 0, 0], version: 5 }]);
		// What's on screen stays until the new copy arrives.
		expect(first(layer, "0/0/0")).toBe(5);
		// Its change event, arriving while that copy loads, doesn't start another.
		layer.changed([{ key: [0, 0, 0], version: 5 }]);
		await tick();
		expect(first(layer, "0/0/0")).toBe(0);
		expect(layer.versionOf("0/0/0")).toBe(5);
		expect(loads).toHaveLength(2);
	});

	it("tells views once a reload has landed", async () => {
		const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
		const { pool } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		const heard = told(layer);
		server.set("0/0/0", { value: 2, version: 2 });
		layer.changed([{ key: [0, 0, 0], version: 2 }]);
		expect(heard).toEqual([]);
		await tick();
		expect(heard).toEqual([["0/0/0"]]);
	});

	it("tells views when a reload stops for good, so they load the chunk afresh", async () => {
		const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		release();
		await loading;
		const heard = told(layer);
		// Someone edits the chunk; while it reloads, the view zooms out past the label limit.
		server.set("0/0/0", { value: 2, version: 2 });
		layer.changed([{ key: [0, 0, 0], version: 2 }]);
		expect(loads).toHaveLength(2);
		layer.store.want("view", []);
		await tick();
		// The old copy is gone, and views drop what they drew from it.
		expect(layer.store.peek("0/0/0")).toBeUndefined();
		expect(heard).toEqual([["0/0/0"]]);
		// Back in view, it loads afresh.
		const again = loaded(layer, "0/0/0");
		release();
		expect((await again).version).toBe(2);
	});

	it("says nothing when a reload is taken over by another, which says so when it lands", async () => {
		const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		release();
		await loading;
		const heard = told(layer);
		layer.reload(["0/0/0"]);
		layer.reload(["0/0/0"]);
		await tick();
		expect(loads).toHaveLength(3);
		// The first reload was replaced, not stopped: the copy stays, and nothing is said yet.
		expect(heard).toEqual([]);
		expect(layer.store.peek("0/0/0")).toBeDefined();
		server.set("0/0/0", { value: 2, version: 2 });
		release();
		await tick();
		expect(heard).toEqual([["0/0/0"]]);
		expect(first(layer, "0/0/0")).toBe(1);
	});

	it("has views load a chunk it edits without a copy of it", () => {
		const { pool } = fakePool(new Map());
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const heard = told(layer);
		layer.applyLocal("op", [delta(5)]);
		expect(heard).toEqual([["0/0/0"]]);
	});

	it("doesn't restart the reload an answered edit started when the edit's change event comes", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		release();
		await loading;
		layer.applyLocal("op", [delta(5)]);
		// Someone else's edit and ours landed: the answer sends for the chunk again.
		server.set("0/0/0", { value: 5, version: 7 });
		layer.settle("op", [{ key: [0, 0, 0], version: 7 }]);
		expect(loads).toHaveLength(2);
		layer.changed([{ key: [0, 0, 0], version: 7 }]);
		expect(loads).toHaveLength(2);
		release();
		await tick();
		expect(layer.versionOf("0/0/0")).toBe(7);
		expect(first(layer, "0/0/0")).toBe(5);
	});

	it("lets go of an answered edit once the load it waited on is cancelled", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		loaded(layer, "0/0/0").catch(() => {});
		layer.applyLocal("op", [delta(5)]);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		// The view moves away: the load the edit waited on stops.
		layer.store.want("view", []);
		await tick();
		// Nothing waits on the chunk any more: a copy that somehow lacks the
		// edit (a later load always has it) is left as it came.
		const again = loaded(layer, "0/0/0");
		release();
		await again;
		await tick();
		expect(first(layer, "0/0/0")).toBe(0);
		expect(layer.versionOf("0/0/0")).toBe(3);
	});

	it("lets go of an answered edit at once when nothing is loading the chunk", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 2 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		// No view shows the chunk now.
		layer.store.want("view", []);
		layer.applyLocal("op", [delta(5)]);
		// Someone else's edit came between: the copy is behind, so it would load again, but nothing shows it.
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		await tick();
		expect(loads).toHaveLength(1);
		expect(layer.store.peek("0/0/0")).toBeUndefined();
		// A later load starts after the answer, so it has the edit; one that doesn't is left as it came.
		const again = loaded(layer, "0/0/0");
		await again;
		expect(first(layer, "0/0/0")).toBe(0);
	});

	it("loads a copy of a version not known once more, showing the edit on it", async () => {
		const server = new Map<string, { value: number; version?: number }>([["0/0/0", { value: 0 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		release();
		await loading;
		expect(first(layer, "0/0/0")).toBe(5);
		await tick();
		expect(loads).toHaveLength(2);
		// The second copy's version isn't known either: the edit goes on it, and that's all.
		release();
		await tick();
		await tick();
		expect(first(layer, "0/0/0")).toBe(5);
		expect(loads).toHaveLength(2);
	});

	it("loads a copy of a version not known once more when its edit is answered", async () => {
		const server = new Map<string, { value: number; version?: number }>([["0/0/0", { value: 0 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		expect(first(layer, "0/0/0")).toBe(5);
		expect(loads).toHaveLength(2);
	});

	it("loads again a copy newer than the edit that was answered, which the edit went over twice", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(5)]);
		// Our edit (4) and someone's after it (5) reached the server, and a copy
		// read after both lands before the answer, with our edit put on top again.
		server.set("0/0/0", { value: 7, version: 5 });
		layer.changed([{ key: [0, 0, 0], version: 5 }]);
		await tick();
		expect(first(layer, "0/0/0")).toBe(5);
		expect(loads).toHaveLength(2);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		expect(loads).toHaveLength(3);
		await tick();
		expect(first(layer, "0/0/0")).toBe(7);
		expect(layer.versionOf("0/0/0")).toBe(5);
	});

	describe("strict edits", () => {
		it("go out on the version your own edit made, even after another edit of yours was refused", async () => {
			const server = new Map([["0/0/0", { value: 0, version: 2 }]]);
			const { pool, release } = fakePool(server, true);
			const layer = new LabelLayer("p", pool, [2, 2, 2]);
			const loading = loaded(layer, "0/0/0");
			layer.applyLocal("op1", [delta(5)]);
			server.set("0/0/0", { value: 5, version: 3 });
			layer.settle("op1", [{ key: [0, 0, 0], version: 3 }]);
			release();
			await loading;
			await tick();
			// A later edit is refused, and its chunk loads again.
			layer.applyLocal("op2", [delta(7)]);
			layer.settle("op2", null);
			await tick();
			const sent = strictOn(layer, { strict: true, deltas: [delta(9)] });
			expect(sent.strict).toBe(true);
			// The server is at 3, our own edit's version.
			expect(sent.deltas[0]?.base_version).toBe(3);
		});

		it("stay strict, on what you saw, when someone else changed the chunk between two of yours, so the server refuses them", async () => {
			const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
			const { pool } = fakePool(server);
			const layer = new LabelLayer("p", pool, [2, 2, 2]);
			await loaded(layer, "0/0/0");
			layer.applyLocal("op", [delta(5)]);
			// Someone else's edit (4), then ours (5).
			const serverVersion = 5;
			server.set("0/0/0", { value: 5, version: serverVersion });
			layer.settle("op", [{ key: [0, 0, 0], version: serverVersion }]);
			const sent = strictOn(layer, { strict: true, deltas: [delta(9)] });
			expect(sent.strict).toBe(true);
			// Based on 3, where the server is at 5: it refuses, and the polygon is redrawn over what changed.
			expect(sent.deltas[0]?.base_version).toBe(3);
			expect(sent.deltas[0]?.base_version).not.toBe(serverVersion);
		});

		it("go out as plain edits where a chunk's version isn't known", async () => {
			const { pool } = fakePool(new Map());
			const layer = new LabelLayer("p", pool, [2, 2, 2]);
			const edit = { strict: true, deltas: [delta(9)] };
			expect(strictOn(layer, edit).strict).toBe(false);
		});

		it("are left as they are when they're not strict", async () => {
			const { pool } = fakePool(new Map());
			const layer = new LabelLayer("p", pool, [2, 2, 2]);
			const edit = { strict: false, deltas: [delta(9)] };
			expect(strictOn(layer, edit)).toBe(edit);
		});
	});

	it("remembers the chunks it edited, the latest last", () => {
		const { pool } = fakePool(new Map());
		const layer = new LabelLayer("p", pool, [128, 128, 128]);
		layer.applyLocal("a", [delta(1, [0, 0, 0]), delta(1, [0, 0, 1])]);
		layer.applyLocal("b", [delta(2, [0, 0, 0])]);
		expect(layer.recent).toEqual(["0/0/1", "0/0/0"]);
	});

	it("drops chunks no view shows instead of refreshing them", async () => {
		const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.store.want("view", new Set());
		const changed: string[][] = [];
		layer.onChange((ids) => changed.push(ids));
		layer.reload(["0/0/0"]);
		expect(layer.store.get("0/0/0")).toBeUndefined();
		expect(changed).toEqual([["0/0/0"]]);
		expect(loads).toHaveLength(1);
	});

	it("refetches a refused edit's chunks", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 2 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(9)]);
		layer.settle("op", null);
		expect(loads).toHaveLength(2);
		await new Promise((r) => setTimeout(r, 0));
		expect(first(layer, "0/0/0")).toBe(0);
	});

	it("leaves a load already under way to bring a refused edit's chunk back clean", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 2 }]]);
		const { pool, loads, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		release();
		await loading;
		// Someone else's edit has the chunk loading again when our edit is refused.
		server.set("0/0/0", { value: 3, version: 3 });
		layer.changed([{ key: [0, 0, 0], version: 3 }]);
		layer.applyLocal("op", [delta(9)]);
		expect(first(layer, "0/0/0")).toBe(9);
		layer.settle("op", null);
		expect(loads).toHaveLength(2);
		release();
		await tick();
		// The load that was under way landed clean, without the refused edit.
		expect(first(layer, "0/0/0")).toBe(3);
		expect(loads).toHaveLength(2);
	});

	describe("a reload that fails", () => {
		afterEach(() => {
			vi.useRealTimers();
		});

		/** A chunk loaded at version 1 whose reload, for someone's edit (2), fails `failures` times. */
		const loadedWithFailures = async (failures: number, held = false) => {
			vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
			const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
			const pool = fakePool(server, held);
			const layer = new LabelLayer("p", pool.pool, [2, 2, 2]);
			const loading = loaded(layer, "0/0/0");
			pool.release();
			await loading;
			server.set("0/0/0", { value: 2, version: 2 });
			pool.failures.left = failures;
			layer.changed([{ key: [0, 0, 0], version: 2 }]);
			pool.release();
			await settled();
			return { layer, ...pool };
		};

		it("leaves the copy shown, and is tried again", async () => {
			const { layer, loads } = await loadedWithFailures(1);
			expect(loads).toHaveLength(2);
			expect(first(layer, "0/0/0")).toBe(1);
			expect(layer.store.peek("0/0/0")).toBeDefined();
			const heard = told(layer);
			await vi.advanceTimersByTimeAsync(1000);
			await settled();
			expect(loads).toHaveLength(3);
			expect(first(layer, "0/0/0")).toBe(2);
			expect(layer.versionOf("0/0/0")).toBe(2);
			expect(heard).toEqual([["0/0/0"]]);
		});

		it("waits twice as long each time, up to half a minute", async () => {
			const { layer, loads, failures } = await loadedWithFailures(6);
			const waits = [1000, 2000, 4000, 8000, 16_000, 30_000];
			for (const [attempt, wait] of waits.entries()) {
				expect(loads).toHaveLength(2 + attempt);
				await vi.advanceTimersByTimeAsync(wait - 1);
				await settled();
				expect(loads).toHaveLength(2 + attempt);
				await vi.advanceTimersByTimeAsync(1);
				await settled();
			}
			// The sixth try worked.
			expect(failures.left).toBe(0);
			expect(loads).toHaveLength(8);
			expect(first(layer, "0/0/0")).toBe(2);
		});

		it("isn't tried again once a reload of it has worked", async () => {
			const { layer, loads } = await loadedWithFailures(1);
			// Another change event reloads the chunk before the retry is due.
			layer.reload(["0/0/0"]);
			await settled();
			expect(loads).toHaveLength(3);
			expect(first(layer, "0/0/0")).toBe(2);
			// The retry's timer is gone with it.
			expect(vi.getTimerCount()).toBe(0);
			await vi.advanceTimersByTimeAsync(60_000);
			await settled();
			expect(loads).toHaveLength(3);
		});

		it("isn't tried again once a newer copy came another way", async () => {
			const { layer, loads } = await loadedWithFailures(1);
			await layer.store.refresh("0/0/0");
			expect(loads).toHaveLength(3);
			await vi.advanceTimersByTimeAsync(60_000);
			await settled();
			expect(loads).toHaveLength(3);
		});

		it("isn't tried again while the chunk is loading another way", async () => {
			const { layer, loads, release } = await loadedWithFailures(1, true);
			const loading = layer.store.refresh("0/0/0");
			expect(loads).toHaveLength(3);
			await vi.advanceTimersByTimeAsync(1000);
			await settled();
			expect(loads).toHaveLength(3);
			release();
			await loading;
		});

		it("isn't tried again once the chunk's copy is gone", async () => {
			const { layer, loads } = await loadedWithFailures(1);
			layer.store.invalidate("0/0/0");
			await vi.advanceTimersByTimeAsync(60_000);
			await settled();
			expect(loads).toHaveLength(2);
		});

		it("isn't tried again after the layer stops", async () => {
			const { layer, loads } = await loadedWithFailures(1);
			layer.stop();
			await vi.advanceTimersByTimeAsync(60_000);
			await settled();
			expect(loads).toHaveLength(2);
		});

		it("keeps what's shown through a failure that isn't followed by a retry", async () => {
			const { layer } = await loadedWithFailures(1);
			expect(first(layer, "0/0/0")).toBe(1);
			expect(layer.versionOf("0/0/0")).toBe(1);
		});
	});
});

describe("adding a class", () => {
	afterEach(() => vi.mocked(api).mockReset());

	const bone = { value: 2, name: "bone", color: "#ff0000" };
	const matrix = { value: 3, name: "matrix", color: "#00ff00" };

	it("adds it to the project, keeps it with the others, and tells the views once the list has it", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		layer.classes = [bone];
		// The POST answers with the class; the list read after it has it too.
		vi.mocked(api).mockResolvedValueOnce(matrix).mockResolvedValueOnce([bone, matrix]);
		const colorsSeen: (string | undefined)[] = [];
		layer.onClasses(() => colorsSeen.push(layer.colors.get(3)));

		expect(await layer.addClass("matrix", "#00ff00")).toEqual(matrix);

		expect(api).toHaveBeenCalledWith("/api/projects/p1/labels/classes", { body: { name: "matrix", color: "#00ff00" } });
		expect(layer.classes.map((c) => c.value)).toEqual([2, 3]);
		// Told after the list had the new color, and only once.
		expect(colorsSeen).toEqual(["#00ff00"]);
	});

	it("keeps the new class even when reading the list again fails", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		layer.classes = [bone];
		vi.mocked(api).mockResolvedValueOnce(matrix).mockRejectedValueOnce(new Error("offline"));
		let told = 0;
		layer.onClasses(() => told++);

		await layer.addClass("matrix", "#00ff00");

		expect(layer.classes).toEqual([bone, matrix]);
		expect(told).toBe(1);
	});

	it("brings in classes others added meanwhile", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		layer.classes = [bone];
		const other = { value: 3, name: "other", color: "#3e63dd" };
		const mine = { value: 4, name: "mine", color: "#f2c14e" };
		vi.mocked(api).mockResolvedValueOnce(mine).mockResolvedValueOnce([bone, other, mine]);

		await layer.addClass("mine", "#f2c14e");

		expect(layer.classes.map((c) => c.name)).toEqual(["bone", "other", "mine"]);
		expect(layer.colors.get(3)).toBe("#3e63dd");
	});

	it("changes nothing when the server refuses", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		let told = 0;
		layer.onClasses(() => told++);
		vi.mocked(api).mockRejectedValue(new Error("No."));

		await expect(layer.addClass("bone", "#ff0000")).rejects.toThrow("No.");

		expect(layer.classes).toEqual([]);
		expect(told).toBe(0);
	});

	it("tells the views only when the classes changed", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		layer.classes = [bone];
		let told = 0;
		layer.onClasses(() => told++);
		vi.mocked(api).mockResolvedValueOnce([bone]).mockResolvedValueOnce([bone, matrix]);

		await layer.refreshClasses();
		expect(told).toBe(0);
		await layer.refreshClasses();
		expect(told).toBe(1);
		expect(layer.classes).toEqual([bone, matrix]);
	});

	it("keeps telling the other views when one throws", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		const quiet = vi.spyOn(console, "error").mockImplementation(() => {});
		let told = 0;
		layer.onClasses(() => {
			throw new Error("a view broke");
		});
		layer.onClasses(() => told++);
		vi.mocked(api).mockResolvedValueOnce(bone).mockResolvedValueOnce([bone]);

		// The class was made, so adding it succeeds however the views took it.
		await expect(layer.addClass("bone", "#ff0000")).resolves.toEqual(bone);

		expect(told).toBe(1);
		quiet.mockRestore();
	});

	it("stops telling views that were removed, or once the layer stops", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		let removed = 0;
		let stopped = 0;
		const stop = layer.onClasses(() => removed++);
		layer.onClasses(() => stopped++);
		stop();
		vi.mocked(api).mockResolvedValueOnce(bone).mockResolvedValueOnce([bone]);
		await layer.addClass("bone", "#ff0000");
		expect([removed, stopped]).toEqual([0, 1]);

		layer.stop();
		vi.mocked(api).mockResolvedValueOnce(matrix).mockResolvedValueOnce([bone, matrix]);
		await layer.addClass("matrix", "#00ff00");
		expect(stopped).toBe(1);
	});
});
