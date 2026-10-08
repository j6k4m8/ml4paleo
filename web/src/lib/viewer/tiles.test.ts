import { describe, expect, it } from "vitest";
import {
	aspectOf,
	boxOnPlane,
	CHUNK,
	chooseLevel,
	countTiles,
	coveringTile,
	type Level,
	PLANES,
	tileCrosses,
	tileId,
	tilesShown,
	tilesToLoad,
	type View,
	viewLevel,
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
		expect(viewLevel(levels, wide, 1000).index).toBe(0);
		expect(viewLevel(levels, wide, 60).index).toBe(1);
		expect(viewLevel(levels, wide, 20).index).toBe(2);
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
		const tiles = tilesToLoad(levels, levels[0] as Level, view({ width: 256, height: 256 }));
		const order = tiles.map((t) => t.level);
		expect(order[0]).toBe(2);
		expect(order.indexOf(0)).toBeGreaterThan(order.lastIndexOf(1));
		const shown = visibleTiles(levels[0] as Level, view({ width: 256, height: 256 }), 0).map(tileId);
		const level0 = tiles.filter((t) => t.level === 0).map(tileId);
		expect(level0.slice(0, shown.length)).toEqual(shown);
		expect(level0.length).toBeGreaterThan(shown.length);
		// Each coarser level covers the view.
		expect(new Set(tiles.filter((t) => t.level === 1).map(tileId))).toEqual(
			new Set(visibleTiles(levels[1] as Level, view({ width: 256, height: 256 }), 0).map(tileId)),
		);
	});

	it("asks for only the coarsest level when that is the one shown", () => {
		const coarsest = levels[2] as Level;
		expect(tilesToLoad(levels, coarsest, view({ zoom: 0.25 })).every((t) => t.level === 2)).toBe(true);
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
