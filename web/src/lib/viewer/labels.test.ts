import { describe, expect, it } from "vitest";
import { base64, packBits, zstdFrame } from "../labels/deltas";
import type { Chunk } from "./chunks";
import { LabelLayer } from "./labels";
import type { WorkerPool } from "./loader";

/**
 * A worker pool whose label chunks come from a map the test controls. With
 * `held`, loads answer from the map as it is when they start but arrive
 * only when the test calls `release`.
 */
function fakePool(server: Map<string, { value: number; version: number }>, held = false) {
	const loads: string[] = [];
	const waiting: (() => void)[] = [];
	const pool = {
		load: async (request: { region: [number, number][] }): Promise<Chunk> => {
			const id = request.region.map(([start]) => start / 64).join("/");
			loads.push(id);
			const { value, version } = server.get(id) ?? { value: 0, version: 0 };
			if (held) await new Promise<void>((resolve) => waiting.push(resolve));
			return { data: new Uint8Array(8).fill(value), shape: [2, 2, 2], version };
		},
	};
	const release = () => {
		for (const resolve of waiting.splice(0)) resolve();
	};
	return { pool: pool as unknown as WorkerPool, loads, release };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

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

	it("puts an edit back on a copy that started loading before the edit applied", async () => {
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
		// The copy predates the edit: it shows the edit anyway, and loads again.
		expect(first(layer, "0/0/0")).toBe(5);
		await tick();
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
		const { pool } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.applyLocal("op", [delta(9)]);
		layer.settle("op", null);
		await new Promise((r) => setTimeout(r, 0));
		expect(first(layer, "0/0/0")).toBe(0);
	});
});
