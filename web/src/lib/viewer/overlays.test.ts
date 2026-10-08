import { describe, expect, it, vi } from "vitest";
import { labelId } from "./labels";
import { labelsView, MAX_LABEL_TILES, MAX_OVERLAY_TILES, overlayHidden } from "./overlays";
import { countDrawn, countTiles, type Level, PLANES, sliceIndex, TILE_HYSTERESIS, tileId, tilesToLoad, type View, viewLevel, visibleTiles } from "./tiles";

vi.mock("#lib/api.ts", () => ({ api: vi.fn() }));

// The labels of a 1024 × 32768 × 32768 image: a pyramid of four levels.
const levels: Level[] = [
	{ index: 0, path: "class", shape: [1024, 32768, 32768], scale: [1, 1, 1] },
	{ index: 1, path: "class_1", shape: [512, 16384, 16384], scale: [2, 2, 2] },
	{ index: 2, path: "class_2", shape: [256, 8192, 8192], scale: [4, 4, 4] },
	{ index: 3, path: "class_3", shape: [128, 4096, 4096], scale: [8, 8, 8] },
];
const fullOnly = levels.slice(0, 1);

function view(over: Partial<View> = {}): View {
	return {
		plane: PLANES.xy,
		position: [512, 16384, 16384],
		zoom: 1,
		aspect: [1, 1, 1],
		width: 640,
		height: 480,
		...over,
	};
}

const noneRecent = { hidden: false, recent: [] as string[], cached: [] as string[] };

describe("overlayHidden", () => {
	// Chunks a view of the full resolution level shows, by its width.
	const wide = (width: number) => view({ width, height: 64 });

	it("is for views that need more than the limit's chunks", () => {
		const level = levels[0] as Level;
		const wideEnough = [...Array(2000).keys()].map((w) => wide(w * 8 + 64)).find((v) => countTiles(level, v) > MAX_OVERLAY_TILES)!;
		expect(overlayHidden(level, wideEnough, false)).toBe(true);
		expect(overlayHidden(level, view(), false)).toBe(false);
	});

	it("stays hidden until the view needs TILE_HYSTERESIS times fewer", () => {
		const level = levels[0] as Level;
		const counts = [...Array(2000).keys()].map((w) => ({ at: wide(w * 8 + 64), count: countTiles(level, wide(w * 8 + 64)) }));
		const over = counts.find((c) => c.count > MAX_OVERLAY_TILES)!;
		const between = counts.find((c) => c.count <= MAX_OVERLAY_TILES && c.count > MAX_OVERLAY_TILES / TILE_HYSTERESIS)!;
		const under = counts.find((c) => c.count <= MAX_OVERLAY_TILES / TILE_HYSTERESIS)!;
		expect(overlayHidden(level, over.at, true)).toBe(true);
		expect(overlayHidden(level, between.at, false)).toBe(false);
		expect(overlayHidden(level, between.at, true)).toBe(true);
		expect(overlayHidden(level, under.at, true)).toBe(false);
	});
});

describe("labelsView", () => {
	it("shows the level that fits, as the image does, with the coarser levels under it, coarsest loading first", () => {
		const at = view({ zoom: 0.5 });
		const shown = labelsView(levels, at, noneRecent);
		expect(shown.level.index).toBe(viewLevel(levels, at, undefined, MAX_LABEL_TILES).index);
		expect(shown.level.index).toBeGreaterThan(0);
		expect(shown.hidden).toBe(false);
		const { tiles, shown: count } = tilesToLoad(levels, shown.level, at);
		expect(shown.wanted).toEqual(tiles.map(labelId));
		expect(shown.shown).toEqual(tiles.slice(0, count).map(labelId));
		// What loads first is the coarsest level.
		expect(shown.wanted[0]?.startsWith(`${levels.length - 1}/`)).toBe(true);
		// What is drawn is what is shown: every level from the one chosen up, finest first.
		expect(shown.draw.map((t) => t.id).sort()).toEqual([...shown.shown].sort());
		const order = shown.draw.map((t) => t.key.level);
		expect(order).toEqual([...order].sort((a, b) => a - b));
		expect(new Set(order)).toEqual(new Set(levels.slice(shown.level.index).map((l) => l.index)));
	});

	it("says which slice and what voxel size each chunk has, as its level does", () => {
		const at = view({ position: [301, 2000, 2100], zoom: 0.25 });
		const { draw } = labelsView(levels, at, noneRecent);
		expect(new Set(draw.map((t) => t.key.level)).size).toBeGreaterThan(1);
		for (const tile of draw) {
			const level = levels[tile.key.level] as Level;
			expect(tile.slice).toBe(sliceIndex(level, at));
			expect(tile.scale).toEqual(level.scale);
		}
	});

	it("names full resolution chunks cz/cy/cx and the coarser ones level/cz/cy/cx", () => {
		const { draw } = labelsView(levels, view({ zoom: 1, width: 128, height: 128 }), noneRecent);
		const ids = draw.map((t) => t.id);
		expect(ids).toContain(visibleTiles(levels[0] as Level, view({ width: 128, height: 128 }), 0).map((k) => `${k.cz}/${k.cy}/${k.cx}`)[0]);
		expect(ids.filter((id) => id.split("/").length === 4).every((id) => Number(id.split("/")[0]) > 0)).toBe(true);
		expect(ids.some((id) => id.split("/").length === 3)).toBe(true);
	});

	it("shows labels where the old limit would have left them out", () => {
		const crowded = view({ zoom: 1, width: 2000, height: 1500 });
		expect(countTiles(levels[0] as Level, crowded)).toBeGreaterThan(MAX_OVERLAY_TILES);
		const shown = labelsView(levels, crowded, noneRecent);
		expect(shown.hidden).toBe(false);
		expect(shown.draw.length).toBeGreaterThan(0);
		expect(countDrawn(levels, shown.level, crowded)).toBeLessThanOrEqual(MAX_LABEL_TILES);
		// Without coarser levels they are left out.
		const alone = labelsView(fullOnly, crowded, noneRecent);
		expect(alone.hidden).toBe(true);
		expect(alone.draw).toEqual([]);
	});

	it("leaves labels out only where even the coarsest level is more than the limit", () => {
		const huge = view({ zoom: 1, width: 40_000, height: 40_000 });
		const shown = labelsView(levels, huge, noneRecent);
		expect(countTiles(levels[3] as Level, huge)).toBeGreaterThan(MAX_LABEL_TILES);
		expect(shown.level.index).toBe(3);
		expect(shown.hidden).toBe(true);
		expect(shown.wanted).toEqual([]);
		expect(shown.draw).toEqual([]);
	});

	it("keeps labels hidden or shown near the limit until the view needs TILE_HYSTERESIS times fewer chunks", () => {
		// With only full resolution chunks, as with a prediction.
		const level = levels[0] as Level;
		const widths = [...Array(4000).keys()].map((w) => w * 4 + 64);
		const at = (width: number) => view({ width, height: 64 });
		const over = at(widths.find((w) => countTiles(level, at(w)) > MAX_OVERLAY_TILES)!);
		const between = at(widths.find((w) => countTiles(level, at(w)) <= MAX_OVERLAY_TILES && countTiles(level, at(w)) > MAX_OVERLAY_TILES / TILE_HYSTERESIS)!);
		expect(labelsView(fullOnly, over, noneRecent).hidden).toBe(true);
		expect(labelsView(fullOnly, between, noneRecent).hidden).toBe(false);
		expect(labelsView(fullOnly, between, { ...noneRecent, hidden: true }).hidden).toBe(true);
	});

	it("keeps the level it showed a little past the point it would leave it, as the image does", () => {
		const at = view({ zoom: 0.8 });
		const first = labelsView(levels, at, { ...noneRecent, current: 1 }).level.index;
		const fresh = labelsView(levels, at, noneRecent).level.index;
		expect(first).toBe(viewLevel(levels, at, 1, MAX_LABEL_TILES).index);
		expect(fresh).toBe(viewLevel(levels, at, undefined, MAX_LABEL_TILES).index);
		expect(first).toBeGreaterThan(fresh);
	});

	describe("this page's latest edits", () => {
		// The chunk at the center of the view, at full resolution (z 512 is chunk 8; y, x 16384 are chunk 256).
		const centered = "8/256/256";

		it("show at full resolution under a coarse level, and load first", () => {
			const at = view({ zoom: 0.25 });
			const shown = labelsView(levels, at, { ...noneRecent, recent: [centered] });
			expect(shown.level.index).toBeGreaterThan(0);
			expect(shown.wanted[0]).toBe(centered);
			expect(shown.shown[0]).toBe(centered);
			expect(shown.draw[0]?.id).toBe(centered);
			expect(shown.draw[0]?.key.level).toBe(0);
			// The levels are there as they were.
			expect(labelsView(levels, at, noneRecent).draw.map((t) => t.id)).toEqual(shown.draw.slice(1).map((t) => t.id));
		});

		it("are shown by themselves where the labels are left out", () => {
			const huge = view({ zoom: 1, width: 40_000, height: 40_000 });
			const shown = labelsView(levels, huge, { ...noneRecent, recent: [centered] });
			expect(shown.hidden).toBe(true);
			expect(shown.wanted).toEqual([centered]);
			expect(shown.shown).toEqual([centered]);
			expect(shown.draw.map((t) => t.id)).toEqual([centered]);
			const alone = labelsView(fullOnly, view({ width: 40_000, height: 40_000 }), { ...noneRecent, recent: [centered] });
			expect(alone.draw.map((t) => t.id)).toEqual([centered]);
		});

		it("are left out where the view doesn't show them: another slice, or off the screen", () => {
			const at = view({ zoom: 0.25 });
			const recent = ["30/256/256", "8/0/0", "8/256/256"];
			const shown = labelsView(levels, at, { ...noneRecent, recent });
			expect(shown.level.index).toBeGreaterThan(0);
			expect(shown.draw.filter((t) => t.key.level === 0).map((t) => t.id)).toEqual(["8/256/256"]);
			expect(shown.wanted.filter((id) => id.split("/").length === 3)).toEqual(["8/256/256"]);
		});

		it("are drawn once where they are also the level the view shows", () => {
			const at = view({ zoom: 1, width: 128, height: 128 });
			const shown = labelsView(levels, at, { ...noneRecent, recent: [centered] });
			expect(shown.level.index).toBe(0);
			expect(shown.draw.filter((t) => t.id === centered)).toHaveLength(1);
			expect(shown.wanted.filter((id) => id === centered)).toHaveLength(1);
		});

		it("number at most as many as the limit for a layer without coarser levels", () => {
			const huge = view({ zoom: 1, width: 40_000, height: 40_000 });
			const recent = [...Array(MAX_OVERLAY_TILES + 40).keys()].map((i) => `8/${244 + (i % 24)}/${244 + Math.floor(i / 24)}`);
			expect(labelsView(fullOnly, huge, { ...noneRecent, recent }).draw).toHaveLength(MAX_OVERLAY_TILES);
		});
	});

	describe("finer chunks at hand", () => {
		const at = view({ zoom: 0.25 });
		const fine = ["8/256/256", "8/256/257", "1/4/128/128"];

		it("show under the level's, and aren't asked for", () => {
			const shown = labelsView(levels, at, { ...noneRecent, cached: fine });
			expect(shown.level.index).toBeGreaterThan(1);
			const drawn = shown.draw.map((t) => t.id);
			expect(drawn.slice(0, 3)).toEqual(expect.arrayContaining(fine.filter((id) => drawn.includes(id))));
			expect(drawn).toContain("8/256/256");
			expect(drawn.indexOf("8/256/256")).toBeLessThan(drawn.findIndex((id) => id.split("/").length === 4 && Number(id.split("/")[0]) === shown.level.index));
			for (const id of fine) {
				expect(shown.wanted).not.toContain(id);
				expect(shown.shown).not.toContain(id);
			}
		});

		it("draw finest first", () => {
			const order = labelsView(levels, at, { ...noneRecent, cached: fine }).draw.map((t) => t.key.level);
			expect(order).toEqual([...order].sort((a, b) => a - b));
		});

		it("don't include chunks of coarser levels, of levels it lacks, or out of view", () => {
			const shown = labelsView(levels, view({ zoom: 1, width: 128, height: 128 }), {
				...noneRecent,
				cached: ["9/0/0/0", "3/0/0/0", "8/0/0", "8/256/256"],
			});
			expect(shown.level.index).toBe(0);
			// Nothing is finer than full resolution, and these aren't in view.
			const drawn = shown.draw.map((t) => t.id);
			for (const id of ["9/0/0/0", "3/0/0/0", "8/0/0"]) expect(drawn).not.toContain(id);
			expect(drawn.filter((id) => id === "8/256/256")).toHaveLength(1);
			// Where the level is coarser, a chunk of a coarser one at hand isn't added to the view's.
			const coarse = labelsView(levels, view({ zoom: 0.25 }), { ...noneRecent, cached: ["3/0/0/0"] });
			expect(coarse.draw.map((t) => t.id)).not.toContain("3/0/0/0");
		});

		it("aren't drawn where the labels are left out", () => {
			const huge = view({ zoom: 1, width: 40_000, height: 40_000 });
			expect(labelsView(levels, huge, { ...noneRecent, cached: fine }).draw).toEqual([]);
		});
	});

	it("draws a chunk once however many ways it comes to be drawn", () => {
		const at = view({ zoom: 0.25 });
		const shown = labelsView(levels, at, { ...noneRecent, recent: ["8/256/256"], cached: ["8/256/256", "8/256/256"] });
		const ids = shown.draw.map((t) => t.id);
		expect(new Set(ids).size).toBe(ids.length);
		expect(shown.wanted.length).toBe(new Set(shown.wanted).size);
	});

	it("names the chunks of a level the same however it was asked for", () => {
		const key = visibleTiles(levels[2] as Level, view({ zoom: 0.25 }), 0)[0]!;
		expect(labelId(key)).toBe(tileId(key));
	});
});
