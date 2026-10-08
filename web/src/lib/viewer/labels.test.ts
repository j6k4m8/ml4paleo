import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "#lib/api.ts";
import { base64, packBits, zstdFrame } from "../labels/deltas";
import type { Chunk } from "./chunks";
import { LabelLayer } from "./labels";
import type { WorkerPool } from "./loader";

/** A worker pool whose label chunks come from a map the test controls. */
function fakePool(server: Map<string, { value: number; version: number }>) {
	const loads: string[] = [];
	const pool = {
		load: async (request: { region: [number, number][] }): Promise<Chunk> => {
			const id = request.region.map(([start]) => start / 64).join("/");
			loads.push(id);
			const { value, version } = server.get(id) ?? { value: 0, version: 0 };
			return { data: new Uint8Array(8).fill(value), shape: [2, 2, 2], version };
		},
	};
	return { pool: pool as unknown as WorkerPool, loads };
}

vi.mock("#lib/api.ts", () => ({ api: vi.fn() }));

const delta = (value: number) => ({
	key: [0, 0, 0] as [number, number, number],
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

	it("advances versions only in sequence, and reloads chunks others changed too", async () => {
		const server = new Map([["0/0/0", { value: 0, version: 3 }]]);
		const { pool, loads } = fakePool(server);
		const layer = new LabelLayer("p", pool, [2, 2, 2]);
		await loaded(layer, "0/0/0");
		layer.noteVersions([{ key: [0, 0, 0], version: 4 }]);
		expect(layer.versionOf("0/0/0")).toBe(4);
		server.set("0/0/0", { value: 1, version: 7 });
		layer.noteVersions([{ key: [0, 0, 0], version: 7 }]);
		await new Promise((r) => setTimeout(r, 0));
		expect(loads).toEqual(["0/0/0", "0/0/0"]);
		expect(layer.versionOf("0/0/0")).toBe(7);
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
