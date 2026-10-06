import { describe, expect, it } from "vitest";
import {
	aspectOf,
	CHUNK,
	chooseLevel,
	type Level,
	PLANES,
	tileId,
	type View,
	visibleTiles,
	voxelAt,
	windowed,
} from "./tiles";

const levels: Level[] = [
	{ index: 0, shape: [256, 1000, 700], scale: [1, 1, 1] },
	{ index: 1, shape: [128, 500, 350], scale: [2, 2, 2] },
	{ index: 2, shape: [64, 250, 175], scale: [4, 4, 4] },
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

	it("stretches thick axes on screen", () => {
		const level = levels[0] as Level;
		// z voxels 4 times as thick: 128 pixels cover 32 z voxels.
		const tiles = visibleTiles(level, view({ plane: PLANES.xz, aspect: [4, 1, 1] }), 0);
		expect(new Set(tiles.map((t) => t.cz))).toEqual(new Set([1, 2]));
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
