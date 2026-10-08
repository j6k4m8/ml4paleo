import { describe, expect, it } from "vitest";
import { CLOSE_PIXELS, closesAt, type Point, type Scale } from "./polygon";

describe("closesAt", () => {
	const triangle: Point[] = [
		[10, 10],
		[40, 10],
		[25, 40],
	];

	it("closes on a click within a few pixels of the first point", () => {
		// Two CSS pixels a voxel: 3 voxels off is 6 pixels, 5 voxels 10.
		const scale: Scale = [2, 2];
		expect(closesAt(triangle, [10, 10], scale)).toBe(true);
		expect(closesAt(triangle, [13, 10], scale)).toBe(true);
		expect(closesAt(triangle, [12, 12], scale)).toBe(true);
		expect(closesAt(triangle, [15, 10], scale)).toBe(false);
		expect(closesAt(triangle, [40, 10], scale)).toBe(false);
	});

	it("measures on screen, so zooming in leaves less room in voxels", () => {
		expect(closesAt(triangle, [13, 10], [1, 1])).toBe(true);
		expect(closesAt(triangle, [13, 10], [4, 4])).toBe(false);
		// Zoomed far out, a click many voxels away still lands on it.
		expect(closesAt(triangle, [40, 10], [0.25, 0.25])).toBe(true);
	});

	it("follows the voxels' shape", () => {
		// Voxels four times as long along v as along u.
		const scale: Scale = [2, 8];
		expect(closesAt(triangle, [10, 11], scale)).toBe(true);
		expect(closesAt(triangle, [10, 11.5], scale)).toBe(false);
		expect(closesAt(triangle, [11.5, 10], scale)).toBe(true);
	});

	it("needs three points first", () => {
		expect(closesAt(triangle.slice(0, 2), [10, 10], [1, 1])).toBe(false);
		expect(closesAt([], [10, 10], [1, 1])).toBe(false);
	});

	it("takes a wider reach", () => {
		expect(closesAt(triangle, [10 + CLOSE_PIXELS * 1.5, 10], [1, 1])).toBe(false);
		expect(closesAt(triangle, [10 + CLOSE_PIXELS * 1.5, 10], [1, 1], CLOSE_PIXELS * 2)).toBe(true);
	});
});
