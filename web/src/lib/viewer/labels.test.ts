import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "#lib/api.ts";
import { base64, packBits, zstdFrame } from "../labels/deltas";
import type { Chunk } from "./chunks";
import { CACHE_BYTES, coarseIds, LabelLayer, labelId, labelKey, labelLevels, strictOn } from "./labels";
import { LoadError, type WorkerPool } from "./loader";
import type { Vec3 } from "./tiles";

/**
 * A worker pool whose label chunks come from a map the test controls. With
 * `held`, loads answer from the map as it is when they start but arrive
 * only when the test calls `release`. The next `failures.left` loads fail,
 * as if the server answered with `failures.status`, or, with none, as if
 * the network failed (`failures.decode`: or what it sent couldn't be read).
 */
function fakePool(server: Map<string, { value: number; version?: number }>, held = false) {
	const loads: string[] = [];
	const waiting: (() => void)[] = [];
	const failures: { left: number; status?: number; decode?: boolean } = { left: 0 };
	const pool = {
		load: async (request: { region: [number, number][] }): Promise<Chunk> => {
			const id = request.region.map(([start]) => start / 64).join("/");
			loads.push(id);
			const { value, version } = server.get(id) ?? { value: 0, version: 0 };
			if (held) await new Promise<void>((resolve) => waiting.push(resolve));
			if (failures.left > 0) {
				failures.left -= 1;
				if (failures.status !== undefined) throw new LoadError("Error: Unexpected response status", failures.status);
				throw new LoadError(failures.decode ? "Error: Invalid zstd frame" : "TypeError: Failed to fetch");
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

	it("doesn't let a copy that had an edit put back over later changes stand for their version", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, release } = fakePool(server, true);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		const loading = loaded(layer, "0/0/0");
		release();
		await loading;
		layer.applyLocal("op", [delta(5)]);
		// Our edit (4), then someone's over it (5). A copy read after both
		// lands before our answer, and our edit is put back over theirs.
		server.set("0/0/0", { value: 7, version: 5 });
		layer.changed([{ key: [0, 0, 0], version: 5 }]);
		release();
		await tick();
		expect(first(layer, "0/0/0")).toBe(5);
		layer.settle("op", [{ key: [0, 0, 0], version: 4 }]);
		// The next strict edit goes out now, before the chunk loads again. It
		// can't be based on 5, which the server would accept though the page
		// shows our 5s where the server has their 7s: on 4, it is refused.
		const sent = strictOn(layer, { strict: true, deltas: [delta(9)] });
		expect(sent.strict).toBe(true);
		expect(sent.deltas[0]?.base_version).toBe(4);
		release();
		await tick();
		expect(layer.versionOf("0/0/0")).toBe(5);
		expect(first(layer, "0/0/0")).toBe(7);
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
		// The waits are random by a quarter either way; this is the middle.
		const random = vi.spyOn(Math, "random");
		beforeEach(() => {
			random.mockReturnValue(0.5);
		});
		afterEach(() => {
			random.mockReset();
			vi.useRealTimers();
		});

		/**
		 * A chunk loaded at version 1 whose reload, for someone's edit (2),
		 * fails `failures` times, with the HTTP `status` if there's one.
		 */
		const loadedWithFailures = async (failures: number, held = false, status?: number, decode = false) => {
			vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
			const server = new Map([["0/0/0", { value: 1, version: 1 }]]);
			const pool = fakePool(server, held);
			const layer = new LabelLayer("p", pool.pool, [2, 2, 2]);
			const loading = loaded(layer, "0/0/0");
			pool.release();
			await loading;
			server.set("0/0/0", { value: 2, version: 2 });
			pool.failures.left = failures;
			pool.failures.status = status;
			pool.failures.decode = decode;
			layer.changed([{ key: [0, 0, 0], version: 2 }]);
			pool.release();
			await settled();
			return { layer, server, ...pool };
		};

		/** Let `ms` pass, a little at a time, so the tries that come of it all run. */
		const wait = async (ms: number) => {
			for (let left = ms; left > 0; left -= 500) {
				await vi.advanceTimersByTimeAsync(Math.min(500, left));
				await settled();
			}
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

		it("is tried again after a wait a quarter shorter or longer, at random", async () => {
			random.mockReturnValueOnce(0);
			const quick = await loadedWithFailures(1);
			await vi.advanceTimersByTimeAsync(749);
			await settled();
			expect(quick.loads).toHaveLength(2);
			await vi.advanceTimersByTimeAsync(1);
			await settled();
			expect(quick.loads).toHaveLength(3);
			vi.useRealTimers();
			random.mockReturnValueOnce(1);
			const slow = await loadedWithFailures(1);
			await vi.advanceTimersByTimeAsync(1249);
			await settled();
			expect(slow.loads).toHaveLength(2);
			await vi.advanceTimersByTimeAsync(1);
			await settled();
			expect(slow.loads).toHaveLength(3);
		});

		it("spreads chunks that failed together over different times", async () => {
			vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
			const server = new Map([
				["0/0/0", { value: 1, version: 1 }],
				["0/0/1", { value: 1, version: 1 }],
			]);
			const { pool, loads, failures } = fakePool(server);
			const layer = new LabelLayer("p", pool, [2, 2, 128]);
			layer.store.want("view", new Set(["0/0/0", "0/0/1"]));
			await Promise.all([layer.store.request("0/0/0"), layer.store.request("0/0/1")]);
			server.set("0/0/0", { value: 2, version: 2 });
			server.set("0/0/1", { value: 2, version: 2 });
			failures.left = 2;
			random.mockReturnValueOnce(0).mockReturnValueOnce(1);
			layer.changed([
				{ key: [0, 0, 0], version: 2 },
				{ key: [0, 0, 1], version: 2 },
			]);
			await settled();
			expect(loads).toHaveLength(4);
			await vi.advanceTimersByTimeAsync(750);
			await settled();
			expect(loads).toHaveLength(5);
			await vi.advanceTimersByTimeAsync(500);
			await settled();
			expect(loads).toHaveLength(6);
		});

		it("isn't tried again when the server says you're signed out or not allowed", async () => {
			for (const status of [401, 403]) {
				const { loads } = await loadedWithFailures(1, false, status);
				expect(vi.getTimerCount()).toBe(0);
				await wait(60_000);
				expect(loads).toHaveLength(2);
				vi.useRealTimers();
			}
		});

		it("isn't tried again for answers that a try won't change", async () => {
			const { loads } = await loadedWithFailures(1, false, 400);
			expect(vi.getTimerCount()).toBe(0);
			await wait(60_000);
			expect(loads).toHaveLength(2);
		});

		it("is tried again when the server is busy, or fails", async () => {
			for (const status of [429, 503]) {
				const { layer, loads } = await loadedWithFailures(1, false, status);
				await wait(1000);
				expect(loads).toHaveLength(3);
				expect(first(layer, "0/0/0")).toBe(2);
				vi.useRealTimers();
			}
		});

		it("stops trying again after a server error eight times in a row", async () => {
			const { loads, failures } = await loadedWithFailures(100, false, 500);
			// The waits: 1, 2, 4, 8, 16, 30, 30 s, then no more.
			await wait((1 + 2 + 4 + 8 + 16 + 30 + 30) * 1000 - 1);
			expect(loads).toHaveLength(8);
			await wait(1);
			expect(loads).toHaveLength(9);
			expect(failures.left).toBe(100 - 8);
			expect(vi.getTimerCount()).toBe(0);
			await wait(10 * 60_000);
			expect(loads).toHaveLength(9);
		});

		it("stops trying again after what the server sent couldn't be read eight times in a row", async () => {
			const { loads, failures } = await loadedWithFailures(100, false, undefined, true);
			await wait((1 + 2 + 4 + 8 + 16 + 30 + 30) * 1000 - 1);
			expect(loads).toHaveLength(8);
			await wait(1);
			expect(loads).toHaveLength(9);
			expect(failures.left).toBe(100 - 8);
			expect(vi.getTimerCount()).toBe(0);
		});

		it("goes on trying again while the network is down, far past the tries a server error gets", async () => {
			const { layer, loads, failures } = await loadedWithFailures(40);
			// Every 30 s once the waits have grown to that.
			await wait(60 * 60_000);
			expect(failures.left).toBe(0);
			expect(loads).toHaveLength(1 + 40 + 1);
			expect(first(layer, "0/0/0")).toBe(2);
		});

		it("stops trying again for good once live updates stop", async () => {
			const sources: { readyState: number; onerror: (() => void) | null }[] = [];
			vi.stubGlobal(
				"EventSource",
				class {
					static CLOSED = 2;
					readyState = 1;
					onerror: (() => void) | null = null;
					constructor() {
						sources.push(this);
					}
					addEventListener() {}
					close() {}
				},
			);
			try {
				vi.mocked(api).mockResolvedValueOnce([]).mockResolvedValueOnce([]);
				const { layer, server, loads, failures } = await loadedWithFailures(1);
				await layer.start();
				let stopped = 0;
				layer.onStopped = () => stopped++;
				// The stream closes for good: signed out, or out of the project.
				sources[0]!.readyState = 2;
				sources[0]!.onerror?.();
				expect(stopped).toBe(1);
				expect(vi.getTimerCount()).toBe(0);
				await wait(60_000);
				expect(loads).toHaveLength(2);
				// And a reload that fails now isn't tried again.
				server.set("0/0/0", { value: 3, version: 3 });
				failures.left = 1;
				layer.changed([{ key: [0, 0, 0], version: 3 }]);
				await settled();
				expect(vi.getTimerCount()).toBe(0);
			} finally {
				vi.unstubAllGlobals();
				vi.mocked(api).mockReset();
			}
		});

		it("starts its waits over for a chunk it stopped trying again because nothing shows it", async () => {
			const { layer, server, loads, failures } = await loadedWithFailures(1);
			// Nothing shows the chunk when the try comes, which drops its copy.
			layer.store.want("view", []);
			await wait(1000);
			expect(layer.store.peek("0/0/0")).toBeUndefined();
			expect(vi.getTimerCount()).toBe(0);
			// Shown again later, it loads, then a reload of it fails: the wait is the first again.
			await loaded(layer, "0/0/0");
			server.set("0/0/0", { value: 3, version: 3 });
			failures.left = 1;
			layer.changed([{ key: [0, 0, 0], version: 3 }]);
			await settled();
			const failed = loads.length;
			await wait(999);
			expect(loads).toHaveLength(failed);
			await wait(1);
			expect(loads).toHaveLength(failed + 1);
		});

		it("keeps what's shown through a failure that isn't followed by a retry", async () => {
			const { layer } = await loadedWithFailures(1);
			expect(first(layer, "0/0/0")).toBe(1);
			expect(layer.versionOf("0/0/0")).toBe(1);
		});
	});
});

describe("counting labels", () => {
	afterEach(() => vi.mocked(api).mockReset());

	it("reads the voxels labeled with each value, background (1) included", async () => {
		const layer = new LabelLayer("p1", fakePool(new Map()).pool, [4, 4, 4]);
		vi.mocked(api).mockResolvedValueOnce({ "1": 0, "2": 1234 });

		const counts = await layer.counts();

		expect(api).toHaveBeenCalledWith("/api/projects/p1/labels/counts");
		expect([...counts]).toEqual([
			[1, 0],
			[2, 1234],
		]);
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

/** What the label zarr's group says of its levels: one `[shape, factor]` each, finest first. */
const group = (levels: [Vec3, Vec3][]) => ({
	zarr_format: 3,
	node_type: "group",
	attributes: {
		ml4paleo: {
			label_levels: levels.map(([shape, factor_zyx], i) => ({ array: i === 0 ? "class" : `class_${i}`, shape, factor_zyx })),
		},
	},
});

// A 256 × 256 × 256 image: 4 × 4 × 4 chunks, then 2 × 2 × 2, then one.
const pyramid: [Vec3, Vec3][] = [
	[[256, 256, 256], [1, 1, 1]],
	[[128, 128, 128], [2, 2, 2]],
	[[64, 64, 64], [4, 4, 4]],
];

describe("the label chunk cache", () => {
	// Chunks of 256 KiB that take no memory to make.
	const big = { byteLength: 256 * 1024, length: 256 * 1024 } as unknown as Uint8Array;

	it("holds 256 MiB of chunks, the three views' share of them, and drops the least recently used past that", async () => {
		expect(CACHE_BYTES).toBe(256 * 1024 * 1024);
		const pool = { load: async (): Promise<Chunk> => ({ data: big, shape: [64, 64, 64] }) } as unknown as WorkerPool;
		const layer = new LabelLayer("p", pool, [64 * 2000, 64, 64]);
		const chunks = 1024;
		await Promise.all(Array.from({ length: chunks }, (_, i) => layer.store.request(`${i}/0/0`)));
		expect(layer.store.bytes).toBe(CACHE_BYTES);
		expect(layer.store.peek("0/0/0")).toBeDefined();
		await layer.store.request(`${chunks}/0/0`);
		expect(layer.store.bytes).toBe(CACHE_BYTES);
		expect(layer.store.peek("0/0/0")).toBeUndefined();
		expect(layer.store.peek("1/0/0")).toBeDefined();
	});
});

describe("labelLevels", () => {
	const shape: Vec3 = [256, 256, 256];

	it("lists the levels the label zarr says it has, finest first", () => {
		expect(labelLevels(group(pyramid), shape)).toEqual([
			{ index: 0, path: "class", shape: [256, 256, 256], scale: [1, 1, 1] },
			{ index: 1, path: "class_1", shape: [128, 128, 128], scale: [2, 2, 2] },
			{ index: 2, path: "class_2", shape: [64, 64, 64], scale: [4, 4, 4] },
		]);
	});

	it("keeps each level's own factors, which may differ by axis", () => {
		const levels = labelLevels(
			group([
				[[256, 256, 256], [1, 1, 1]],
				[[256, 128, 128], [1, 2, 2]],
				[[128, 64, 64], [2, 4, 4]],
			]),
			shape,
		);
		expect(levels.map((l) => l.scale)).toEqual([[1, 1, 1], [1, 2, 2], [2, 4, 4]]);
	});

	it("stays at full resolution when the server lists no levels, or something else", () => {
		const full = [{ index: 0, path: "class", shape, scale: [1, 1, 1] }];
		for (const found of [undefined, null, [], "no", { zarr_format: 3 }, { attributes: {} }, { attributes: { ml4paleo: {} } }, { attributes: { ml4paleo: { label_levels: [] } } }]) {
			expect(labelLevels(found, shape)).toEqual(full);
		}
	});

	it("stays at full resolution when the levels are for another image", () => {
		const full = [{ index: 0, path: "class", shape, scale: [1, 1, 1] }];
		expect(labelLevels(group([[[128, 256, 256], [1, 1, 1]]]), shape)).toEqual(full);
		expect(labelLevels(group([[[256, 256, 256], [2, 2, 2]]]), shape)).toEqual(full);
	});

	it("stops at the first level it can't make sense of", () => {
		const bad: unknown[] = [
			{ array: "class_2", shape: [64, 64, 64], factor_zyx: [4, 4, 4] },
			{ array: "class_1", shape: [128, 128], factor_zyx: [2, 2, 2] },
			{ array: "class_1", shape: [128, 128, 128], factor_zyx: [0, 2, 2] },
			{ array: "class_1", shape: [128, 128, 128], factor_zyx: [1.5, 2, 2] },
			// No coarser than full resolution.
			{ array: "class_1", shape: [256, 256, 256], factor_zyx: [1, 1, 1] },
			null,
		];
		for (const entry of bad) {
			const found = {
				attributes: {
					ml4paleo: { label_levels: [{ array: "class", shape: [256, 256, 256], factor_zyx: [1, 1, 1] }, entry, { array: "class_2", shape: [64, 64, 64], factor_zyx: [4, 4, 4] }] },
				},
			};
			expect(labelLevels(found, shape).map((l) => l.path)).toEqual(["class"]);
		}
		// A coarser level that's finer along an axis than the one before isn't one.
		const crossed = group([
			[[256, 256, 256], [1, 1, 1]],
			[[128, 128, 128], [2, 2, 2]],
			[[256, 64, 64], [1, 4, 4]],
		]);
		expect(labelLevels(crossed, shape).map((l) => l.path)).toEqual(["class", "class_1"]);
	});

	describe("a level's shape", () => {
		const paths = (found: unknown, of: Vec3) => labelLevels(found, of).map((l) => l.path);

		it("is the image's divided by its factor and rounded up, as the server plans it, so a level of an odd-sized image is a voxel larger than half", () => {
			// What `plan_levels` gives for 257 × 257 × 257 voxels.
			const odd: Vec3 = [257, 257, 257];
			const planned: [Vec3, Vec3][] = [
				[odd, [1, 1, 1]],
				[[129, 129, 129], [2, 2, 2]],
				[[65, 65, 65], [4, 4, 4]],
				[[33, 33, 33], [8, 8, 8]],
			];
			expect(labelLevels(group(planned), odd).map((l) => [l.shape, l.scale])).toEqual(planned);
		});

		it("is checked: a level as big as its image, or rounded down, is no level of it, and the levels before it are kept", () => {
			const odd: Vec3 = [257, 257, 257];
			// Listed at the image's own size with a factor of two: its chunks would reach past the array the server has.
			expect(paths(group([[odd, [1, 1, 1]], [odd, [2, 2, 2]]]), odd)).toEqual(["class"]);
			expect(paths(group([[odd, [1, 1, 1]], [[128, 128, 128], [2, 2, 2]]]), odd)).toEqual(["class"]);
			// A later level that is wrong ends the levels there.
			const third = (size: Vec3) =>
				group([
					[odd, [1, 1, 1]],
					[[129, 129, 129], [2, 2, 2]],
					[size, [4, 4, 4]],
				]);
			expect(paths(third([65, 65, 65]), odd)).toEqual(["class", "class_1", "class_2"]);
			for (const wrong of [[64, 64, 64], [66, 66, 66], [129, 129, 129], [65, 65, 64], [65, 64, 65], [64, 65, 65]] as Vec3[]) {
				expect(paths(third(wrong), odd), String(wrong)).toEqual(["class", "class_1"]);
			}
		});

		it("is checked along each axis by its own factor, for an odd-sized image whose voxels differ in size", () => {
			// What `plan_levels` gives for 45 × 513 × 511 voxels, 4 × 1 × 1 apart.
			const image: Vec3 = [45, 513, 511];
			const planned: [Vec3, Vec3][] = [
				[image, [1, 1, 1]],
				[[45, 257, 256], [1, 2, 2]],
				[[45, 129, 128], [1, 4, 4]],
				[[23, 65, 64], [2, 8, 8]],
				[[12, 33, 32], [4, 16, 16]],
			];
			expect(labelLevels(group(planned), image).map((l) => [l.shape, l.scale])).toEqual(planned);
			// Any one axis of any one level off by a voxel ends the levels before it.
			for (let level = 1; level < planned.length; level++) {
				for (let axis = 0; axis < 3; axis++) {
					const off = planned.map(([size, factor], i) => [i === level ? (size.map((n, a) => (a === axis ? n + 1 : n)) as Vec3) : size, factor] as [Vec3, Vec3]);
					expect(paths(group(off), image).length, `level ${level}, axis ${axis}`).toBe(level);
				}
			}
		});
	});
});

describe("label chunk ids", () => {
	it("number full resolution chunks cz/cy/cx and coarser ones level/cz/cy/cx", () => {
		expect(labelId({ level: 0, cz: 3, cy: 1, cx: 2 })).toBe("3/1/2");
		expect(labelId({ level: 2, cz: 0, cy: 1, cx: 0 })).toBe("2/0/1/0");
		expect(labelKey("3/1/2")).toEqual({ level: 0, cz: 3, cy: 1, cx: 2 });
		expect(labelKey("2/0/1/0")).toEqual({ level: 2, cz: 0, cy: 1, cx: 0 });
	});
});

/**
 * A worker pool for labels with several levels, whose chunks come from a map
 * the test controls, by the ids `labelId` makes. Levels in `absent` have no
 * array: the server answers 404 for it. With `held`, loads answer from the
 * map as it is when they start but arrive only when the test calls `release`.
 * The next `failures.left` loads fail, as if the server answered with
 * `failures.status`.
 */
function levelPool(chunks: Map<string, { value: number; version?: number; pyramid?: number }> = new Map(), absent = new Set<number>(), held = false) {
	const loads: string[] = [];
	const asked: { path: string; derived?: boolean }[] = [];
	const waiting: (() => void)[] = [];
	const failures: { left: number; status: number } = { left: 0, status: 500 };
	const pool = {
		load: async (request: { path: string; region: [number, number][]; derived?: boolean }, signal?: AbortSignal): Promise<Chunk> => {
			const level = request.path === "class" ? 0 : Number(request.path.slice("class_".length));
			const [z, y, x] = request.region.map(([start]) => start / 64);
			const id = labelId({ level, cz: z!, cy: y!, cx: x! });
			loads.push(id);
			asked.push({ path: request.path, derived: request.derived });
			if (absent.has(level)) throw new LoadError("NotFoundError: Not found: v3 array or group", 404);
			const { value, version, pyramid } = chunks.get(id) ?? { value: 0 };
			if (held) {
				// A load cancelled while it waits ends at once, as the pool's does.
				await new Promise<void>((resolve, reject) => {
					waiting.push(resolve);
					signal?.addEventListener("abort", () => reject(new DOMException("No longer needed", "AbortError")), { once: true });
				});
			}
			if (failures.left > 0) {
				failures.left -= 1;
				throw new LoadError(`Error: Unexpected response status ${failures.status}`, failures.status);
			}
			return { data: new Uint8Array(8).fill(value), shape: [2, 2, 2], version, pyramid };
		},
	};
	const release = () => {
		for (const resolve of waiting.splice(0)) resolve();
	};
	return { pool: pool as unknown as WorkerPool, loads, asked, chunks, failures, release };
}

/** A stand-in for the browser's EventSource that the test can send changes through. */
class FakeSource {
	static last: FakeSource | undefined;
	static CLOSED = 2;
	readyState = 1;
	onerror: (() => void) | null = null;
	#handlers = new Map<string, (event: { data: string }) => void>();
	constructor(readonly url: string) {
		FakeSource.last = this;
	}
	addEventListener(type: string, handler: (event: { data: string }) => void) {
		this.#handlers.set(type, handler);
	}
	close() {}
	/** The server says these chunks changed. */
	change(chunks: { key: Vec3; version: number }[]) {
		this.#handlers.get("change")?.({ data: JSON.stringify({ chunks }) });
	}
}

/** A layer for a 256-cubed image that has started against a server answering `zarr` for its group. */
async function started(zarr: unknown, pooled = levelPool()) {
	vi.stubGlobal("EventSource", FakeSource);
	vi.mocked(api).mockResolvedValueOnce([]).mockResolvedValueOnce([]);
	if (zarr instanceof Error) vi.mocked(api).mockRejectedValueOnce(zarr);
	else vi.mocked(api).mockResolvedValueOnce(zarr);
	const layer = new LabelLayer("p", pooled.pool, [256, 256, 256]);
	await layer.start();
	return { layer, source: FakeSource.last!, ...pooled };
}

describe("label levels", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.mocked(api).mockReset();
		FakeSource.last = undefined;
	});

	it("start with full resolution only, until the server says what it has", () => {
		const layer = new LabelLayer("p", levelPool().pool, [256, 256, 256]);
		expect(layer.levels.map((l) => l.path)).toEqual(["class"]);
	});

	it("are read from the group's metadata when the layer starts", async () => {
		const { layer } = await started(group(pyramid));
		expect(layer.levels.map((l) => [l.path, l.scale])).toEqual([
			["class", [1, 1, 1]],
			["class_1", [2, 2, 2]],
			["class_2", [4, 4, 4]],
		]);
		expect(vi.mocked(api).mock.calls.map(([path]) => path)).toEqual([
			"/api/projects/p/labels/classes",
			"/api/projects/p/labels/ops?limit=1",
			"/api/projects/p/labels/zarr/zarr.json",
		]);
	});

	it("are only full resolution for a server that doesn't have them", async () => {
		for (const found of [{ zarr_format: 3, node_type: "group", attributes: {} }, new Error("HTTP 404")]) {
			const { layer } = await started(found);
			expect(layer.levels.map((l) => l.path)).toEqual(["class"]);
		}
	});

	it("load a coarser level's chunks from its array, as made ones, and a full resolution chunk from the class array", async () => {
		const { layer, asked } = await started(group(pyramid), levelPool(new Map([["1/0/1/0", { value: 4, pyramid: 9 }]])));
		layer.store.want("view", ["1/0/1/0", "0/0/1/0"]);
		const coarse = await layer.store.request("1/0/1/0");
		await layer.store.request("0/0/1/0");
		expect(asked).toEqual([
			{ path: "class_1", derived: true },
			{ path: "class", derived: false },
		]);
		// What it was made from is no edit's base version.
		expect(coarse.pyramid).toBe(9);
		expect(layer.versionOf("1/0/1/0")).toBeUndefined();
	});

	it("fall back to full resolution when the server has no array for a level it listed, and tell the views", async () => {
		const { layer } = await started(group(pyramid), levelPool(new Map(), new Set([2])));
		const heard = vi.fn();
		layer.onLevels(heard);
		layer.store.want("view", ["2/0/0/0", "1/0/0/0"]);
		// Its load ends without an error, as one nobody needs now does.
		const lacking = await layer.store.request("2/0/0/0").catch((e: unknown) => e);
		expect((lacking as DOMException).name).toBe("AbortError");
		expect(layer.levels.map((l) => l.path)).toEqual(["class", "class_1"]);
		expect(heard).toHaveBeenCalledTimes(1);
		// A load of it again doesn't say so again.
		await layer.store.request("2/0/0/0").catch(() => {});
		expect(heard).toHaveBeenCalledTimes(1);
		await layer.store.request("1/0/0/0");
	});

	it("fall back past every coarser level a server lacks, forgetting what was loaded of them", async () => {
		const { layer } = await started(group(pyramid), levelPool(new Map(), new Set([1])));
		layer.store.want("view", ["0/0/0", "2/0/0/0", "1/0/0/0"]);
		await layer.store.request("0/0/0");
		await layer.store.request("2/0/0/0");
		expect(layer.store.peek("2/0/0/0")).toBeDefined();
		await layer.store.request("1/0/0/0").catch(() => {});
		expect(layer.levels.map((l) => l.path)).toEqual(["class"]);
		expect(layer.store.peek("2/0/0/0")).toBeUndefined();
		expect(layer.store.peek("0/0/0")).toBeDefined();
	});

	it("aren't given up for a failure that isn't a missing array", async () => {
		const failing = levelPool();
		const { layer } = await started(group(pyramid), failing);
		const heard = vi.fn();
		layer.onLevels(heard);
		const load = failing.pool.load;
		failing.pool.load = async (request, signal) => {
			if (request.path === "class_1") throw new LoadError("Error: Unexpected response status 500", 500);
			return load.call(failing.pool, request, signal);
		};
		layer.store.want("view", ["1/0/0/0"]);
		await expect(layer.store.request("1/0/0/0")).rejects.toBeInstanceOf(LoadError);
		expect(layer.levels).toHaveLength(3);
		expect(heard).not.toHaveBeenCalled();
	});

	it("aren't given up for a chunk the server is still making, which is asked for again", async () => {
		vi.useFakeTimers();
		try {
			const making = levelPool(new Map([["1/0/0/0", { value: 6, pyramid: 3 }]]));
			const { layer } = await started(group(pyramid), making);
			making.failures.left = 2;
			making.failures.status = 503;
			const heard = vi.fn();
			layer.onLevels(heard);
			layer.store.want("view", ["1/0/0/0"]);
			const loading = layer.store.request("1/0/0/0");
			// Each time it is asked again after two seconds and up to a second more.
			await vi.advanceTimersByTimeAsync(1999);
			expect(making.loads).toHaveLength(1);
			await vi.advanceTimersByTimeAsync(1001);
			expect(making.loads).toHaveLength(2);
			await vi.advanceTimersByTimeAsync(3000);
			expect((await loading).pyramid).toBe(3);
			expect(making.loads).toHaveLength(3);
			expect(layer.levels).toHaveLength(3);
			expect(heard).not.toHaveBeenCalled();
		} finally {
			vi.useRealTimers();
		}
	});

	it("take only two of the four places for loads at once, the others being for full resolution chunks", async () => {
		const { layer, loads } = await started(group(pyramid), levelPool(new Map(), new Set(), true));
		const coarse = ["2/0/0/0", "1/0/0/0", "1/0/0/1", "1/0/1/0"];
		const full = ["0/0/0", "0/0/1", "0/1/0", "0/1/1"];
		layer.store.want("view", [...coarse, ...full]);
		for (const id of [...coarse, ...full]) layer.store.request(id).catch(() => {});
		expect(loads.filter((id) => labelKey(id).level > 0)).toEqual(["2/0/0/0", "1/0/0/0"]);
		expect(loads.filter((id) => labelKey(id).level === 0)).toEqual(["0/0/0", "0/0/1"]);
	});

	it("take all four places for loads while no full resolution chunk is wanted, as zoomed out views do", async () => {
		const { layer, loads } = await started(group(pyramid), levelPool(new Map(), new Set(), true));
		const coarse = ["2/0/0/0", "1/0/0/0", "1/0/0/1", "1/0/1/0", "1/0/1/1"];
		layer.store.want("view", coarse);
		for (const id of coarse.slice(0, 3)) layer.store.request(id).catch(() => {});
		expect(loads).toEqual(coarse.slice(0, 3));
		for (const id of coarse.slice(3)) layer.store.request(id).catch(() => {});
		expect(loads).toEqual(coarse.slice(0, 4));
	});

	it("stop telling views once the layer stops", async () => {
		const { layer } = await started(group(pyramid), levelPool(new Map(), new Set([1])));
		const heard = vi.fn();
		layer.onLevels(heard);
		layer.stop();
		layer.store.want("view", ["1/0/0/0"]);
		await layer.store.request("1/0/0/0").catch(() => {});
		expect(heard).not.toHaveBeenCalled();
	});
});

describe("coarseIds", () => {
	const levels = labelLevels(group(pyramid), [256, 256, 256]);

	it("numbers the chunk of each coarser level that holds a chunk by its factor", () => {
		expect(coarseIds(levels, [[5, 9, 17]])).toEqual(["1/2/4/8", "2/1/2/4"]);
		expect(coarseIds(levels, [[0, 0, 0]])).toEqual(["1/0/0/0", "2/0/0/0"]);
		expect(coarseIds(levels, [[3, 3, 3]])).toEqual(["1/1/1/1", "2/0/0/0"]);
	});

	it("uses each axis's own factor", () => {
		const stretched = labelLevels(
			group([
				[[256, 256, 256], [1, 1, 1]],
				[[256, 128, 128], [1, 2, 2]],
				[[128, 64, 64], [2, 4, 4]],
			]),
			[256, 256, 256],
		);
		expect(coarseIds(stretched, [[5, 9, 17]])).toEqual(["1/5/4/8", "2/2/2/4"]);
	});

	it("names a chunk held by several only once", () => {
		expect(coarseIds(levels, [[4, 4, 4], [5, 5, 5], [4, 5, 5], [6, 4, 7]])).toEqual(["1/2/2/2", "1/3/2/3", "2/1/1/1"]);
	});

	it("has none without coarser levels", () => {
		expect(coarseIds(levels.slice(0, 1), [[1, 2, 3]])).toEqual([]);
		expect(coarseIds(levels, [])).toEqual([]);
	});
});

describe("coarser label chunks that changed", () => {
	const random = vi.spyOn(Math, "random");
	beforeEach(() => {
		random.mockReturnValue(0.5);
	});
	afterEach(() => {
		random.mockReset();
		vi.useRealTimers();
		vi.unstubAllGlobals();
		vi.mocked(api).mockReset();
		FakeSource.last = undefined;
	});

	// A change at chunk (1, 2, 3) is under these.
	const fine = [1, 2, 3] as Vec3;
	const under = { one: "1/0/1/1", two: "2/0/0/0" };

	/** Two views' worth of coarser chunks under chunk (1, 2, 3), loaded, and shown by a view. */
	const shown = async (held = false) => {
		const chunks = new Map([
			[under.one, { value: 1, pyramid: 4 }],
			[under.two, { value: 1, pyramid: 4 }],
		]);
		const started_ = await started(group(pyramid), levelPool(chunks, new Set(), held));
		const wanted = [under.one, under.two];
		started_.layer.store.want("view", wanted);
		const loading = Promise.all(wanted.map((id) => started_.layer.store.request(id)));
		started_.release();
		await loading;
		started_.loads.length = 0;
		return started_;
	};
	const wait = async (ms: number) => {
		for (let left = ms; left > 0; left -= 500) {
			await vi.advanceTimersByTimeAsync(Math.min(500, left));
			await settled();
		}
	};
	/** The coarser chunks among those views were told of, once each, in order. */
	const coarse = (heard: string[][]) => [...new Set(heard.flat().filter((id) => labelKey(id).level > 0))].sort();
	/** The coarser chunks among those loaded. */
	const coarsely = (loads: string[]) => loads.filter((id) => labelKey(id).level > 0).sort();

	it("are loaded again at every level when a change event names a chunk under them", async () => {
		const { layer, source, loads, chunks } = await shown();
		const heard = told(layer);
		chunks.set(under.one, { value: 2, pyramid: 5 });
		chunks.set(under.two, { value: 3, pyramid: 5 });
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(coarsely(loads)).toEqual([under.one, under.two]);
		expect(first(layer, under.one)).toBe(2);
		expect(first(layer, under.two)).toBe(3);
		expect(layer.store.peek(under.one)?.pyramid).toBe(5);
		// Views drop what they drew from the old copies.
		expect(coarse(heard)).toEqual([under.one, under.two]);
	});

	it("keep the old copy showing until the new one arrives", async () => {
		const { layer, source, release, chunks } = await shown(true);
		chunks.set(under.one, { value: 2, pyramid: 5 });
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(first(layer, under.one)).toBe(1);
		release();
		await tick();
		expect(first(layer, under.one)).toBe(2);
	});

	it("are loaded again when this page's own edit is answered, even with nothing left of the edit to show", async () => {
		const { layer, loads, chunks } = await shown();
		chunks.set(under.one, { value: 2, pyramid: 5 });
		layer.settle("an edit whose copies were all current", [{ key: fine, version: 5 }]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
		expect(first(layer, under.one)).toBe(2);
	});

	it("are loaded again for an edit that is still on screen at full resolution too", async () => {
		const { layer, loads } = await shown();
		layer.applyLocal("op", [delta(5, fine)]);
		expect(loads).toEqual([]);
		layer.settle("op", [{ key: fine, version: 5 }]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
	});

	it("aren't loaded again for an edit the server refused, which changed nothing", async () => {
		const { layer, loads } = await shown();
		layer.applyLocal("op", [delta(5, fine)]);
		layer.settle("op", null);
		await tick();
		expect(loads.filter((id) => id.split("/").length > 3)).toEqual([]);
	});

	it("are loaded again once for an edit that both its answer and its change event name", async () => {
		const { layer, source, loads } = await shown();
		layer.settle("op", [{ key: fine, version: 5 }]);
		await tick();
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
		// The other way round too.
		loads.length = 0;
		source.change([{ key: fine, version: 6 }]);
		await tick();
		layer.settle("op two", [{ key: fine, version: 6 }]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
	});

	it("are loaded again for a later change of the same chunk", async () => {
		const { source, loads } = await shown();
		source.change([{ key: fine, version: 5 }]);
		await tick();
		source.change([{ key: fine, version: 6 }]);
		await tick();
		expect(loads).toHaveLength(4);
	});

	it("start over when a later change comes while they load, which the load may not have", async () => {
		const { layer, source, loads, release, chunks } = await shown(true);
		source.change([{ key: fine, version: 5 }]);
		await tick();
		// Still loading, a second change of the same chunk comes.
		chunks.set(under.one, { value: 2, pyramid: 6 });
		source.change([{ key: fine, version: 6 }]);
		await tick();
		expect(loads.filter((id) => id === under.one)).toHaveLength(2);
		release();
		await tick();
		expect(first(layer, under.one)).toBe(2);
		expect(layer.store.isLoading(under.one)).toBe(false);
	});

	it("don't start over for a change of the same version that they already load for", async () => {
		const { layer, source, loads } = await shown(true);
		source.change([{ key: fine, version: 5 }]);
		await tick();
		layer.settle("op", [{ key: fine, version: 5 }]);
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(loads.filter((id) => id === under.one)).toHaveLength(1);
	});

	it("are loaded again once per change to chunks they share, however many the change names", async () => {
		const { source, loads } = await shown();
		// Chunks (0, 2, 2) .. (1, 3, 3) are all under level 1's (0, 1, 1) and level 2's (0, 0, 0).
		source.change([
			{ key: [0, 2, 2], version: 2 },
			{ key: [0, 3, 3], version: 3 },
			{ key: [1, 2, 3], version: 4 },
			{ key: [1, 3, 2], version: 5 },
		]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
	});

	it("are forgotten, not loaded again, when no view shows them", async () => {
		const { layer, source, loads } = await shown();
		layer.store.want("view", []);
		const heard = told(layer);
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(loads).toEqual([]);
		expect(layer.store.peek(under.one)).toBeUndefined();
		expect(layer.store.peek(under.two)).toBeUndefined();
		expect(coarse(heard)).toEqual([under.one, under.two]);
	});

	it("aren't loaded for chunks the page never loaded", async () => {
		const { source, loads } = await shown();
		source.change([{ key: [3, 3, 3], version: 2 }]);
		await tick();
		// (3, 3, 3) is under level 1's (1, 1, 1), which was never shown, and level 2's, which was.
		expect(loads).toEqual([under.two]);
	});

	it("are all that is loaded again when the full resolution chunk is current already", async () => {
		const { layer, source, loads } = await shown();
		layer.store.want("view", [under.one, under.two, "1/2/3"]);
		await layer.store.request("1/2/3");
		loads.length = 0;
		layer.store.get("1/2/3")!.version = 5;
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(loads.sort()).toEqual([under.one, under.two]);
	});

	it("are loaded again after a failure, as full resolution chunks are", async () => {
		vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
		const { layer, source, loads, failures } = await shown();
		failures.left = 1;
		source.change([{ key: fine, version: 5 }]);
		await settled();
		// Level 1's reload failed, level 2's worked.
		expect(loads.sort()).toEqual([under.one, under.two]);
		expect(layer.store.peek(under.one)).toBeDefined();
		await wait(999);
		expect(loads).toHaveLength(2);
		await wait(1);
		expect(loads.sort()).toEqual([under.one, under.one, under.two]);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("keep the copy shown while the server is still making the new one, which is asked for again without taking a place", async () => {
		vi.useFakeTimers();
		const { layer, source, loads, failures, chunks } = await shown();
		chunks.set(under.two, { value: 2, pyramid: 5 });
		failures.left = 1;
		failures.status = 503;
		const heard = told(layer);
		source.change([{ key: [3, 3, 3], version: 5 }]);
		await settled();
		expect(loads).toEqual([under.two]);
		// It waits to be asked again, the old copy on show, and nobody told of it (of the chunks dropped, they were).
		expect(layer.store.isLoading(under.two)).toBe(true);
		expect(first(layer, under.two)).toBe(1);
		expect(heard.flat()).not.toContain(under.two);
		heard.length = 0;
		await vi.advanceTimersByTimeAsync(2499);
		expect(loads).toHaveLength(1);
		await vi.advanceTimersByTimeAsync(1);
		await settled();
		expect(loads).toEqual([under.two, under.two]);
		expect(first(layer, under.two)).toBe(2);
		expect(heard).toEqual([[under.two]]);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("are asked for again by the layer, the old copy still shown, when the server was still making them after the store's two minutes", async () => {
		vi.useFakeTimers();
		const { layer, source, loads, failures } = await shown();
		failures.left = 1_000_000;
		failures.status = 503;
		source.change([{ key: [3, 3, 3], version: 5 }]);
		await settled();
		// The store asks every 2.5 s, from 0 to 120 s, and then fails with the server still busy.
		await wait(119_999);
		expect(loads).toHaveLength(48);
		await wait(1);
		expect(loads).toHaveLength(49);
		expect(first(layer, under.two)).toBe(1);
		// The layer asks again as it does after any failure.
		await wait(999);
		expect(loads).toHaveLength(49);
		await wait(1);
		expect(loads).toHaveLength(50);
		expect(first(layer, under.two)).toBe(1);
	});

	it("are dropped, not left out of date, once loading them again has failed eight times", async () => {
		vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
		const { layer, source, loads, failures } = await shown();
		failures.left = 1000;
		const heard = told(layer);
		source.change([{ key: [3, 3, 3], version: 5 }]);
		await settled();
		// Level 1's chunk for it was never loaded; level 2's is under.two.
		expect(loads).toEqual([under.two]);
		heard.length = 0;
		await wait((1 + 2 + 4 + 8 + 16 + 30 + 30) * 1000 - 1);
		expect(loads).toHaveLength(7);
		expect(layer.store.peek(under.two)).toBeDefined();
		expect(heard).toEqual([]);
		await wait(1);
		expect(loads).toHaveLength(8);
		// The out-of-date copy is gone, and views are told, so they load it afresh.
		expect(layer.store.peek(under.two)).toBeUndefined();
		expect(heard).toEqual([[under.two]]);
		expect(vi.getTimerCount()).toBe(0);
		failures.left = 0;
		layer.store.want("view", [under.two]);
		expect(((await layer.store.request(under.two)).data as Uint8Array)[0]).toBe(1);
	});

	it("are dropped at once when loading them again is refused", async () => {
		vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
		const { layer, source, loads, failures } = await shown();
		failures.left = 1;
		failures.status = 403;
		const heard = told(layer);
		source.change([{ key: [3, 3, 3], version: 5 }]);
		await settled();
		expect(loads).toEqual([under.two]);
		expect(layer.store.peek(under.two)).toBeUndefined();
		expect(heard.at(-1)).toEqual([under.two]);
		expect(vi.getTimerCount()).toBe(0);
	});

	it("go with a load again that no view wants any more, so the next look loads them afresh", async () => {
		const { layer, source, release, chunks } = await shown(true);
		const heard = told(layer);
		chunks.set(under.two, { value: 4, pyramid: 7 });
		source.change([{ key: [3, 3, 3], version: 5 }]);
		await tick();
		heard.length = 0;
		layer.store.want("view", [under.one]);
		await tick();
		expect(layer.store.peek(under.two)).toBeUndefined();
		expect(heard).toEqual([[under.two]]);
		layer.store.want("view", [under.one, under.two]);
		const again = layer.store.request(under.two);
		release();
		expect(((await again).data as Uint8Array)[0]).toBe(4);
	});

	it("aren't loaded again by anything without coarser levels", async () => {
		const { layer, source, loads } = await started(group(pyramid.slice(0, 1)));
		layer.store.want("view", ["1/2/3"]);
		await layer.store.request("1/2/3");
		loads.length = 0;
		source.change([{ key: fine, version: 5 }]);
		await tick();
		expect(loads).toEqual(["1/2/3"]);
	});

	describe("stay out of what edits are sent on", () => {
		it("whatever version they were made from", async () => {
			const { layer } = await shown();
			expect(layer.store.peek(under.one)?.pyramid).toBe(4);
			expect(layer.versionOf(under.one)).toBeUndefined();
			expect(layer.versionOf(under.two)).toBeUndefined();
		});

		it("so an edit goes out on the full resolution chunk's version, or a plain edit if that isn't known", async () => {
			const { layer } = await shown();
			layer.store.want("view", [under.one, under.two, "1/2/3"]);
			await layer.store.request("1/2/3");
			layer.store.get("1/2/3")!.version = 3;
			const op = { strict: true, deltas: [delta(5, fine)] };
			expect(strictOn(layer, op).deltas[0]!.base_version).toBe(3);
			// Nothing of the coarser chunks above it can stand in for a version it lacks.
			const lacking = { strict: true, deltas: [delta(5, [1, 2, 2])] };
			expect(strictOn(layer, lacking).strict).toBe(false);
		});

		it("and don't make a full resolution chunk's change event seem seen", async () => {
			const { layer, source, loads } = await shown();
			layer.store.want("view", [under.one, under.two, "1/2/3"]);
			await layer.store.request("1/2/3");
			loads.length = 0;
			// The coarser chunk's version (4) is past the change's (3), which tells nothing about the chunk under it.
			source.change([{ key: fine, version: 3 }]);
			await tick();
			expect(loads).toContain("1/2/3");
		});
	});
});
