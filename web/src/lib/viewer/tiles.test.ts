import { describe, expect, it } from "vitest";
import {
	aspectOf,
	boxOnPlane,
	CHUNK,
	chooseLevel,
	countDrawn,
	countTiles,
	coveringTile,
	LEVEL_HYSTERESIS,
	type Level,
	PLANES,
	TILE_HYSTERESIS,
	type TileKey,
	tileCrosses,
	tileId,
	tilesShown,
	tilesToLoad,
	tilesUnder,
	type Vec3,
	type View,
	viewLevel,
	visibleBox,
	visibleTiles,
	voxelAt,
	windowed,
} from "./tiles";

const levels: Level[] = [
	{ index: 0, path: "0", shape: [256, 1000, 700], scale: [1, 1, 1] },
	{ index: 1, path: "1", shape: [128, 500, 350], scale: [2, 2, 2] },
	{ index: 2, path: "2", shape: [64, 250, 175], scale: [4, 4, 4] },
];

function view(over: Partial<View> = {}): View {
	return {
		plane: PLANES.xy,
		position: [130, 500, 350],
		zoom: 1,
		aspect: [1, 1, 1],
		width: 128,
		height: 128,
		...over,
	};
}

describe("chooseLevel", () => {
	it("uses full resolution when zoomed in", () => {
		expect(chooseLevel(levels, view({ zoom: 2 })).index).toBe(0);
		expect(chooseLevel(levels, view({ zoom: 1 })).index).toBe(0);
	});

	it("uses coarser levels as the view zooms out", () => {
		expect(chooseLevel(levels, view({ zoom: 0.5 })).index).toBe(1);
		expect(chooseLevel(levels, view({ zoom: 0.25 })).index).toBe(2);
		expect(chooseLevel(levels, view({ zoom: 0.01 })).index).toBe(2);
	});

	it("keeps the level it shows a little past the switch, zooming in", () => {
		// Level 1's voxels cover 2 × zoom pixels; past 1.5 the normal choice is level 0.
		const past = view({ zoom: 0.8 });
		expect(chooseLevel(levels, past).index).toBe(0);
		expect(chooseLevel(levels, past, 1).index).toBe(1);
		// Far enough past, it moves on.
		const beyond = view({ zoom: (1.5 * LEVEL_HYSTERESIS) / 2 + 0.01 });
		expect(chooseLevel(levels, beyond, 1).index).toBe(0);
		// Zooming out switches to the coarser level at once, and back only past the band.
		expect(chooseLevel(levels, view({ zoom: 0.74 }), 0).index).toBe(1);
		expect(chooseLevel(levels, view({ zoom: 0.76 }), 1).index).toBe(1);
		// A level two steps coarser isn't kept.
		expect(chooseLevel(levels, view({ zoom: 0.8 }), 2).index).toBe(0);
	});

	it("doesn't flip between levels as the zoom wobbles around a switch", () => {
		let shown: number | undefined;
		const seen = new Set<number>();
		for (const zoom of [0.74, 0.76, 0.75, 0.77, 0.79, 0.76, 0.78]) {
			shown = chooseLevel(levels, view({ zoom }), shown).index;
			seen.add(shown);
		}
		expect([...seen]).toEqual([1]);
	});

	it("keeps no coarser level that is no coarser on this plane", () => {
		// Level 1 halves only z, which an XY view doesn't see.
		const flat: Level[] = [
			{ index: 0, path: "0", shape: [256, 100, 100], scale: [1, 1, 1] },
			{ index: 1, path: "1", shape: [128, 100, 100], scale: [2, 1, 1] },
		];
		expect(chooseLevel(flat, view({ zoom: 1 }), 1).index).toBe(0);
	});

	it("counts stretched axes at their physical size", () => {
		// z voxels are 4 times as thick: on an XZ view at zoom 0.5, z already
		// spans 2 pixels per voxel, but x needs level 1.
		const thick = view({ plane: PLANES.xz, aspect: [4, 1, 1], zoom: 0.5 });
		expect(chooseLevel(levels, thick).index).toBe(1);
	});
});

describe("visibleTiles", () => {
	it("covers the view plus padding, nearest the center first", () => {
		const level = levels[0] as Level;
		const tiles = visibleTiles(level, view(), 0);
		// y 436..564 and x 286..414
		expect(new Set(tiles.map(tileId))).toEqual(
			new Set(["0/2/6/4", "0/2/6/5", "0/2/6/6", "0/2/7/4", "0/2/7/5", "0/2/7/6", "0/2/8/4", "0/2/8/5", "0/2/8/6"]),
		);
		expect(tileId(tiles[0] as never)).toBe("0/2/7/5");
		expect(visibleTiles(level, view(), 1)).toHaveLength(25);
	});

	it("maps each plane's axes onto chunk keys", () => {
		const level = levels[0] as Level;
		// XZ through y = 500: chunks along z (vertical) and x (horizontal).
		const xz = visibleTiles(level, view({ plane: PLANES.xz }), 0);
		expect(xz.every((t) => t.cy === Math.floor(500 / CHUNK))).toBe(true);
		expect(new Set(xz.map((t) => t.cz))).toEqual(new Set([1, 2, 3]));
		expect(new Set(xz.map((t) => t.cx))).toEqual(new Set([4, 5, 6]));
		// YZ through x = 350: z horizontal, y vertical.
		const yz = visibleTiles(level, view({ plane: PLANES.yz, width: 256 }), 0);
		expect(yz.every((t) => t.cx === Math.floor(350 / CHUNK))).toBe(true);
		expect(new Set(yz.map((t) => t.cz))).toEqual(new Set([0, 1, 2, 3]));
		expect(new Set(yz.map((t) => t.cy))).toEqual(new Set([6, 7, 8]));
	});

	it("uses the level's own voxels", () => {
		const level = levels[2] as Level;
		const tiles = visibleTiles(level, view({ zoom: 0.25 }), 0);
		expect(tiles.every((t) => t.level === 2 && t.cz === Math.floor(130 / 4 / CHUNK))).toBe(true);
	});

	it("stays inside the image", () => {
		const level = levels[0] as Level;
		const corner = visibleTiles(level, view({ position: [130, 0, 0] }), 2);
		expect(corner.every((t) => t.cy >= 0 && t.cx >= 0)).toBe(true);
		expect(visibleTiles(level, view({ position: [256, 500, 350] }))).toEqual([]);
		expect(visibleTiles(level, view({ position: [-1, 500, 350] }))).toEqual([]);
		const far = visibleTiles(level, view({ position: [130, 10_000, 10_000] }), 1);
		expect(far.every((t) => t.cy <= Math.ceil(1000 / CHUNK) - 1 && t.cx <= Math.ceil(700 / CHUNK) - 1)).toBe(true);
	});

	it("lists the view before its margin", () => {
		const level = levels[0] as Level;
		const shown = new Set(visibleTiles(level, view(), 0).map(tileId));
		const tiles = visibleTiles(level, view(), 1).map(tileId);
		expect(new Set(tiles.slice(0, shown.size))).toEqual(shown);
	});

	it("orders by distance on screen, where thick voxels are tall", () => {
		const level = levels[0] as Level;
		// XZ with z voxels 4 times as thick: a chunk is 256 pixels tall and 64
		// wide, so the next chunk across comes before the next one down.
		const tiles = visibleTiles(level, view({ plane: PLANES.xz, aspect: [4, 1, 1], position: [96, 500, 352], width: 400, height: 600 }), 0);
		expect(tileId(tiles[0] as never)).toBe("0/1/7/5");
		expect(tiles.slice(1, 3).map(tileId).sort()).toEqual(["0/1/7/4", "0/1/7/6"]);
	});

	it("stretches thick axes on screen", () => {
		const level = levels[0] as Level;
		// z voxels 4 times as thick: 128 pixels cover 32 z voxels.
		const tiles = visibleTiles(level, view({ plane: PLANES.xz, aspect: [4, 1, 1] }), 0);
		expect(new Set(tiles.map((t) => t.cz))).toEqual(new Set([1, 2]));
	});
});

describe("viewLevel", () => {
	it("goes coarser when the chosen level would take too many chunks", () => {
		const wide = view({ zoom: 2, width: 2000, height: 1000 });
		// x 0..700 (11 chunks) by y 250..750 (9 chunks), then 6 by 5 at level 1.
		expect(countTiles(levels[0] as Level, wide)).toBe(11 * 9);
		expect(countTiles(levels[1] as Level, wide)).toBe(6 * 5);
		expect(viewLevel(levels, wide, undefined, 1000).index).toBe(0);
		expect(viewLevel(levels, wide, undefined, 60).index).toBe(1);
		expect(viewLevel(levels, wide, undefined, 20).index).toBe(2);
	});

	it("doesn't flip levels as panning near the limit adds and drops a row of chunks", () => {
		const limit = 120;
		let shown: number | undefined;
		const seen = new Set<number>();
		// Level 0 at zoom 2: 1800 × 1000 pixels show 900 × 500 voxels, 11 or 12 chunks
		// across (the image is 700 wide, so x stays at 11) and 8 or 9 down as y moves.
		for (const y of [480, 500, 510, 530, 490, 470, 505, 515]) {
			const at = view({ zoom: 2, width: 1800, height: 1000, position: [130, y, 350] });
			shown = viewLevel(levels, at, shown, limit).index;
			seen.add(shown);
		}
		expect(seen.size).toBe(1);
	});

	it("doesn't flip levels as panning takes the level's own chunks across the limit", () => {
		// Level 0 takes 88 or 99 chunks as y moves, so a limit between them
		// used to switch levels back and forth with every row that came or went.
		const limit = 94;
		let shown: number | undefined;
		const seen = new Set<number>();
		const counts = new Set<number>();
		for (const y of [480, 510, 500, 530, 470, 515, 490, 505]) {
			const at = view({ zoom: 2, width: 1800, height: 1000, position: [130, y, 350] });
			counts.add(countTiles(levels[0] as Level, at));
			shown = viewLevel(levels, at, shown, limit).index;
			seen.add(shown);
		}
		expect([...counts].sort()).toEqual([88, 99]);
		expect(seen.size).toBe(1);
	});

	it("goes finer again once well under the limit", () => {
		const limit = 120;
		const near = view({ zoom: 2, width: 1800, height: 1000, position: [130, 500, 350] });
		const level0 = countDrawn(levels, levels[0] as Level, near);
		expect(level0).toBeGreaterThan(limit / TILE_HYSTERESIS);
		// Kept at level 1 by the limit, it stays there while level 0 would only just fit...
		expect(viewLevel(levels, near, 1, Math.ceil(level0 * 1.1)).index).toBe(1);
		// ...and goes back to level 0 once that fits well under it.
		expect(viewLevel(levels, near, 1, Math.ceil(level0 * TILE_HYSTERESIS)).index).toBe(0);
		// Without a level shown before, it takes level 0 when that fits.
		expect(viewLevel(levels, near, undefined, level0).index).toBe(0);
	});

	it("counts the coarser levels drawn under a level against the limit", () => {
		const wide = view({ zoom: 2, width: 2000, height: 1000 });
		// Level 1's own 30 chunks fit under 35, but not with level 2's 9 under them.
		expect(countDrawn(levels, levels[1] as Level, wide)).toBe(30 + 9);
		expect(viewLevel(levels, wide, undefined, 35).index).toBe(2);
	});
});

describe("countTiles", () => {
	it("counts what visibleTiles lists", () => {
		const level = levels[0] as Level;
		for (const padding of [0, 1, 2]) {
			expect(countTiles(level, view(), padding)).toBe(visibleTiles(level, view(), padding).length);
		}
		expect(countTiles(level, view({ position: [-1, 500, 350] }))).toBe(0);
	});
});

describe("tilesToLoad", () => {
	it("asks for coarser levels first, then the level's view, then its margin", () => {
		const { tiles, shown: drawn } = tilesToLoad(levels, levels[0] as Level, view({ width: 256, height: 256 }));
		const order = tiles.map((t) => t.level);
		expect(order[0]).toBe(2);
		expect(order.indexOf(0)).toBeGreaterThan(order.lastIndexOf(1));
		const shown = visibleTiles(levels[0] as Level, view({ width: 256, height: 256 }), 0).map(tileId);
		const level0 = tiles.filter((t) => t.level === 0).map(tileId);
		expect(level0.slice(0, shown.length)).toEqual(shown);
		expect(level0.length).toBeGreaterThan(shown.length);
		// The ones drawn come first, margin last.
		expect(drawn).toBe(countDrawn(levels, levels[0] as Level, view({ width: 256, height: 256 })));
		expect(tiles.slice(drawn).every((t) => t.level === 0 && !shown.includes(tileId(t)))).toBe(true);
		// Each coarser level covers the view.
		expect(new Set(tiles.filter((t) => t.level === 1).map(tileId))).toEqual(
			new Set(visibleTiles(levels[1] as Level, view({ width: 256, height: 256 }), 0).map(tileId)),
		);
	});

	it("asks for a coarser chunk under every chunk the view shows, on every plane", () => {
		for (const plane of [PLANES.xy, PLANES.xz, PLANES.yz]) {
			for (const [position, zoom, aspect] of [
				[[130, 500, 350], 1, [1, 1, 1]],
				[[3, 999, 2], 2.5, [1, 1, 1]],
				[[200, 37, 650], 0.7, [4, 1, 1]],
				[[64, 640, 128], 1.3, [1, 2, 2]],
			] as [Vec3, number, Vec3][]) {
				const at = view({ plane, position, zoom, aspect, width: 300, height: 200 });
				const level = levels[0] as Level;
				const asked = new Set(tilesToLoad(levels, level, at).tiles.map(tileId));
				for (const key of visibleTiles(level, at, 0)) {
					for (const coarser of levels.slice(1)) expect(asked).toContain(tileId(coveringTile(key, level, coarser, at)));
				}
			}
		}
	});

	it("asks for only the coarsest level when that is the one shown", () => {
		const coarsest = levels[2] as Level;
		expect(tilesToLoad(levels, coarsest, view({ zoom: 0.25 })).tiles.every((t) => t.level === 2)).toBe(true);
	});
});

describe("coveringTile", () => {
	it("finds the coarser chunk holding a chunk's part of the plane", () => {
		const fine = levels[0] as Level;
		const coarse = levels[2] as Level;
		// z 130 is level-2 slice 32, in chunk 0; x chunk 5 (320..383) is in level-2 chunk 1.
		expect(coveringTile({ level: 0, cz: 2, cy: 7, cx: 5 }, fine, coarse, view())).toEqual({ level: 2, cz: 0, cy: 1, cx: 1 });
	});
});

describe("tilesUnder", () => {
	it("finds the finer chunks shown under the missing coarser ones, nearest the center first", () => {
		const fine = levels[0] as Level;
		const coarse = levels[1] as Level;
		const at = view({ width: 256, height: 256 });
		const shown = visibleTiles(coarse, at, 0);
		const missing = new Set([tileId(shown[0] as TileKey)]);
		const under = tilesUnder(fine, coarse, missing, at);
		expect(under.length).toBeGreaterThan(0);
		// Each lies under the one missing chunk, and every such chunk the view shows is there.
		expect(under.every((key) => missing.has(tileId(coveringTile(key, fine, coarse, at))))).toBe(true);
		const all = visibleTiles(fine, at, 0).filter((key) => missing.has(tileId(coveringTile(key, fine, coarse, at))));
		expect(under.map(tileId)).toEqual(all.map(tileId));
		// A missing chunk the view doesn't show has nothing under it, and none missing, nothing.
		expect(tilesUnder(fine, coarse, new Set(["1/0/0/0"]), at)).toEqual([]);
		expect(tilesUnder(fine, coarse, new Set(), at)).toEqual([]);
	});

	it("finds a quarter of each missing chunk's area in the next finer level", () => {
		const fine = levels[0] as Level;
		const coarse = levels[1] as Level;
		// Level 1's chunks cover 128 voxels a side, level 0's 64: four chunks under one.
		const at = view({ width: 1024, height: 1024, zoom: 1 });
		const middle = visibleTiles(coarse, at, 0).filter((key) => key.cy === 3 && key.cx === 3);
		expect(tilesUnder(fine, coarse, new Set(middle.map(tileId)), at)).toHaveLength(4);
	});
});

describe("tilesShown", () => {
	it("keeps the chunks on the view's slice and screen, nearest first", () => {
		const level = levels[0] as Level;
		const keys = [
			{ level: 0, cz: 2, cy: 8, cx: 6 },
			{ level: 0, cz: 2, cy: 7, cx: 5 },
			{ level: 0, cz: 3, cy: 7, cx: 5 },
			{ level: 0, cz: 2, cy: 0, cx: 0 },
		];
		expect(tilesShown(keys, level, view()).map(tileId)).toEqual(["0/2/7/5", "0/2/8/6"]);
	});
});

describe("aspectOf", () => {
	it("is relative to the finest axis", () => {
		expect(aspectOf([4, 2, 2])).toEqual([2, 1, 1]);
		expect(aspectOf(null)).toEqual([1, 1, 1]);
		expect(aspectOf([0, 1, 1])).toEqual([1, 1, 1]);
	});
});

describe("voxelAt", () => {
	it("maps screen offsets to voxels along the plane's axes", () => {
		expect(voxelAt(view({ zoom: 2 }), 10, -4)).toEqual([130, 498, 355]);
		expect(voxelAt(view({ plane: PLANES.yz, aspect: [2, 1, 1] }), 8, 3)).toEqual([134, 503, 350]);
	});
});

describe("windowed", () => {
	it("maps the window onto 0..1", () => {
		expect(windowed(50, 0, 100)).toBe(0.5);
		expect(windowed(-5, 0, 100)).toBe(0);
		expect(windowed(500, 0, 100)).toBe(1);
		expect(windowed(3, 3, 3)).toBe(1);
	});
});

describe("boxOnPlane", () => {
	// z 10..20, y 20..40, x 30..60
	const box = [10, 20, 30, 20, 40, 60];

	it("gives the rectangle a box covers on the planes it crosses", () => {
		expect(boxOnPlane(box, PLANES.xy, 10)).toEqual([30, 20, 60, 40]);
		expect(boxOnPlane(box, PLANES.xz, 39)).toEqual([30, 10, 60, 20]);
		expect(boxOnPlane(box, PLANES.yz, 45)).toEqual([10, 20, 20, 40]);
	});

	it("is null on the planes it misses", () => {
		expect(boxOnPlane(box, PLANES.xy, 9)).toBeNull();
		expect(boxOnPlane(box, PLANES.xy, 20)).toBeNull();
		expect(boxOnPlane(box, PLANES.yz, 60)).toBeNull();
	});
});

describe("tileCrosses", () => {
	const key = (cz: number, cy: number, cx: number) => ({ level: 0, cz, cy, cx });

	it("finds the chunks a rectangle overlaps", () => {
		// x 30..70, y 20..40 on an XY plane: chunks x 0 and 1, y 0.
		const rect: [number, number, number, number] = [30, 20, CHUNK + 6, 40];
		expect(tileCrosses(key(3, 0, 0), PLANES.xy, rect)).toBe(true);
		expect(tileCrosses(key(3, 0, 1), PLANES.xy, rect)).toBe(true);
		expect(tileCrosses(key(3, 0, 2), PLANES.xy, rect)).toBe(false);
		expect(tileCrosses(key(3, 1, 0), PLANES.xy, rect)).toBe(false);
		// On a YZ plane u is z: z 30..70 reaches chunks z 0 and 1.
		expect(tileCrosses(key(1, 0, 5), PLANES.yz, rect)).toBe(true);
		expect(tileCrosses(key(2, 0, 5), PLANES.yz, rect)).toBe(false);
	});
});

describe("visibleBox", () => {
	// A view of an image 100 (z) × 200 (y) × 300 (x) voxels, and where its canvas reaches.
	const SHAPE: Vec3 = [100, 200, 300];

	it("is the voxels the canvas covers on the slice, one voxel thick", () => {
		// 400 × 200 pixels at 4 pixels a voxel: 100 × 50 voxels about x 150, y 100.
		const zoomedIn = view({ position: [10.5, 100, 150], zoom: 4, width: 400, height: 200 });
		expect(visibleBox(zoomedIn, SHAPE)).toEqual([10, 75, 100, 11, 125, 200]);
	});

	it("counts a voxel the canvas covers only in part", () => {
		// x 90.25 to 110.25, y 49.5 to 50.5.
		const partly = view({ position: [3, 50, 100.25], zoom: 10, width: 200, height: 10 });
		expect(visibleBox(partly, SHAPE)).toEqual([3, 49, 90, 4, 51, 111]);
		// A single voxel, at ten pixels a voxel.
		expect(visibleBox(view({ position: [3.5, 50.5, 100.5], zoom: 10, width: 10, height: 10 }), SHAPE)).toEqual([3, 50, 100, 4, 51, 101]);
	});

	it("doesn't take in a voxel the canvas only touches, however the pixels round", () => {
		// Its edges are at x 89.999999999 and 109.999999999, y 40 and 60.
		const edge = view({ position: [3, 50, 100 - 1e-9], zoom: 1, width: 20, height: 20 });
		expect(visibleBox(edge, SHAPE)).toEqual([3, 40, 90, 4, 60, 110]);
	});

	it("is cut to the image where the canvas is bigger", () => {
		const whole = view({ position: [50, 100, 150], zoom: 1, width: 1000, height: 800 });
		expect(visibleBox(whole, SHAPE)).toEqual([50, 0, 0, 51, 200, 300]);
		// Panned to a corner: x -30 to 70, y -15 to 35.
		const corner = view({ position: [0.5, 10, 20], zoom: 2, width: 200, height: 100 });
		expect(visibleBox(corner, SHAPE)).toEqual([0, 0, 0, 1, 35, 70]);
		const far = view({ position: [99.5, 190, 290], zoom: 2, width: 200, height: 100 });
		expect(visibleBox(far, SHAPE)).toEqual([99, 165, 240, 100, 200, 300]);
	});

	it("takes each plane's own axes", () => {
		// 40 × 20 pixels at 2 a voxel: 20 voxels across and 10 up and down.
		const around = { position: [30.5, 40.5, 50.5] as Vec3, zoom: 2, width: 40, height: 20 };
		const shape: Vec3 = [60, 80, 100];
		// XY: x across, y up and down, at z 30.
		expect(visibleBox(view({ ...around, plane: PLANES.xy }), shape)).toEqual([30, 35, 40, 31, 46, 61]);
		// XZ: x across, z up and down, at y 40.
		expect(visibleBox(view({ ...around, plane: PLANES.xz }), shape)).toEqual([25, 40, 40, 36, 41, 61]);
		// YZ: z across, y up and down, at x 50.
		expect(visibleBox(view({ ...around, plane: PLANES.yz }), shape)).toEqual([20, 35, 50, 41, 46, 51]);
	});

	it("shows fewer voxels along an axis whose voxels are thicker", () => {
		// z voxels four times as thick as the rest: each is 8 pixels at zoom 2.
		const aspect: Vec3 = [4, 1, 1];
		const thick = { position: [50.5, 100, 150] as Vec3, zoom: 2, aspect, width: 400, height: 400 };
		const shape: Vec3 = [200, 300, 400];
		// XZ: 400 pixels across is 200 voxels of x, and up and down 50 voxels of z.
		expect(visibleBox(view({ ...thick, plane: PLANES.xz }), shape)).toEqual([25, 100, 50, 76, 101, 250]);
		// XY has no thick axis on screen, and YZ has it across.
		expect(visibleBox(view({ ...thick, plane: PLANES.xy }), shape)).toEqual([50, 0, 50, 51, 200, 250]);
		expect(visibleBox(view({ ...thick, plane: PLANES.yz }), shape)).toEqual([25, 0, 150, 76, 200, 151]);
	});

	it("takes the slice the crosshair is in", () => {
		const at = (z: number) => visibleBox(view({ position: [z, 100, 150], width: 20, height: 20 }), SHAPE)!;
		expect([at(0.5)[0], at(0.99)[0], at(1)[0], at(99.99)[0]]).toEqual([0, 0, 1, 99]);
	});

	it("is null where nothing of the image shows", () => {
		const base = { position: [10.5, 100, 150] as Vec3, width: 100, height: 100 };
		expect(visibleBox(view({ ...base, width: 0 }), SHAPE)).toBeNull();
		expect(visibleBox(view({ ...base, height: 0 }), SHAPE)).toBeNull();
		expect(visibleBox(view({ ...base, zoom: 0 }), SHAPE)).toBeNull();
		expect(visibleBox(view({ ...base, position: [100, 100, 150] }), SHAPE)).toBeNull();
		expect(visibleBox(view({ ...base, position: [-0.5, 100, 150] }), SHAPE)).toBeNull();
		// Off the side of the image.
		expect(visibleBox(view({ ...base, position: [10, 100, 400] }), SHAPE)).toBeNull();
		expect(visibleBox(view({ ...base, position: [10, 100, Number.NaN] }), SHAPE)).toBeNull();
	});
});
