import { describe, expect, it } from "vitest";
import { roiBox, thinAxis, within } from "./rois.svelte";
import { PLANES } from "./viewer/tiles";

describe("roiBox", () => {
	const shape: [number, number, number] = [100, 200, 300];

	it("makes a one-voxel slice on the drawn plane", () => {
		expect(roiBox(PLANES.xy, 40, [[10.2, 20.7], [50.5, 30.1]], 1, shape)).toEqual([40, 20, 10, 41, 31, 51]);
		expect(roiBox(PLANES.yz, 7, [[3, 9], [8, 12]], 1, shape)).toEqual([3, 9, 7, 8, 12, 8]);
	});

	it("centers cubes on the slice and keeps them inside", () => {
		expect(roiBox(PLANES.xy, 40, [[0, 0], [64, 64]], 64, shape)).toEqual([8, 0, 0, 72, 64, 64]);
		expect(roiBox(PLANES.xy, 98, [[0, 0], [64, 64]], 64, shape)).toEqual([36, 0, 0, 100, 64, 64]);
		expect(roiBox(PLANES.xz, 5, [[-10, -10], [400, 400]], 500, shape)).toEqual([0, 0, 0, 100, 200, 300]);
	});

	it("rounds fractional corners outward", () => {
		expect(roiBox(PLANES.xy, 40, [[10.3, 20.7], [12.9, 22.1]], 1, shape)).toEqual([40, 20, 10, 41, 23, 13]);
	});

	it("refuses empty rectangles", () => {
		expect(roiBox(PLANES.xy, 40, [[5, 5], [5, 9]], 1, shape)).toBeNull();
	});
});

describe("thinAxis", () => {
	it("finds the slice axis", () => {
		expect(thinAxis([3, 0, 0, 4, 10, 10])).toBe(0);
		expect(thinAxis([0, 0, 7, 10, 10, 8])).toBe(2);
		expect(thinAxis([0, 0, 0, 10, 10, 10])).toBe(0);
	});
});

describe("within", () => {
	it("is true only for boxes inside the other on every axis", () => {
		const outer = [0, 10, 20, 30, 40, 50];
		expect(within([0, 10, 20, 30, 40, 50], outer)).toBe(true);
		expect(within([5, 15, 25, 6, 30, 45], outer)).toBe(true);
		expect(within([5, 15, 25, 6, 30, 51], outer)).toBe(false);
		expect(within([5, 9, 25, 6, 30, 45], outer)).toBe(false);
	});
});
