import { describe, expect, it } from "vitest";
import { CHUNK, chooseLevel, type Level, tileId, visibleTiles, windowed } from "./tiles";

const levels: Level[] = [
	{ index: 0, shape: [256, 1000, 700], scale: [1, 1, 1] },
	{ index: 1, shape: [128, 500, 350], scale: [2, 2, 2] },
	{ index: 2, shape: [64, 250, 175], scale: [4, 4, 4] },
];

describe("chooseLevel", () => {
	it("uses full resolution when zoomed in", () => {
		expect(chooseLevel(levels, 2).index).toBe(0);
		expect(chooseLevel(levels, 1).index).toBe(0);
	});

	it("uses coarser levels as the view zooms out", () => {
		expect(chooseLevel(levels, 0.5).index).toBe(1);
		expect(chooseLevel(levels, 0.25).index).toBe(2);
		expect(chooseLevel(levels, 0.01).index).toBe(2);
	});
});

describe("visibleTiles", () => {
	const view = { z: 130, centerY: 500, centerX: 350, zoom: 1, width: 128, height: 128 };

	it("covers the view plus padding, nearest the center first", () => {
		const level = levels[0] as Level;
		const tiles = visibleTiles(level, view, 0);
		// 436..564 in y and 286..414 in x
		expect(new Set(tiles.map(tileId))).toEqual(
			new Set(["0/2/6/4", "0/2/6/5", "0/2/6/6", "0/2/7/4", "0/2/7/5", "0/2/7/6", "0/2/8/4", "0/2/8/5", "0/2/8/6"]),
		);
		expect(tileId(tiles[0] as never)).toBe("0/2/7/5");
		expect(visibleTiles(level, view, 1)).toHaveLength(25);
	});

	it("maps the plane into the level's chunks", () => {
		const level = levels[2] as Level;
		const tiles = visibleTiles(level, { ...view, zoom: 0.25 }, 0);
		expect(tiles.every((t) => t.level === 2 && t.cz === Math.floor(130 / 4 / CHUNK))).toBe(true);
	});

	it("stays inside the image", () => {
		const level = levels[0] as Level;
		const tiles = visibleTiles(level, { ...view, centerX: 0, centerY: 0 }, 2);
		expect(tiles.every((t) => t.cy >= 0 && t.cx >= 0)).toBe(true);
		expect(visibleTiles(level, { ...view, z: 256 })).toEqual([]);
		expect(visibleTiles(level, { ...view, z: -1 })).toEqual([]);
		const far = visibleTiles(level, { ...view, centerX: 10_000, centerY: 10_000 }, 1);
		expect(far.every((t) => t.cy <= Math.ceil(1000 / CHUNK) - 1 && t.cx <= Math.ceil(700 / CHUNK) - 1)).toBe(true);
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
