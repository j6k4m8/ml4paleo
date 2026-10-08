import { describe, expect, it } from "vitest";
import { apart, CLOSE_PIXELS, closesAt, LASSO_SPACING, lassoPoints, type Point, type Scale } from "./polygon";
import { PlaneMask } from "./raster";

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

describe("lassoPoints", () => {
	/** A pointer's path around a circle, sampled every `step` voxels. */
	function circle(radius: number, step: number, center: Point = [50, 50]): Point[] {
		const count = Math.ceil((2 * Math.PI * radius) / step);
		return Array.from({ length: count }, (_, i) => {
			const angle = (2 * Math.PI * i) / count;
			return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
		});
	}

	/** Each point's distance on screen from the one before it. */
	function gaps(points: Point[], scale: Scale): number[] {
		return points.slice(1).map((point, i) => apart(points[i]!, point, scale));
	}

	it("drops a point every few pixels of a slow drag", () => {
		const scale: Scale = [1, 1];
		const samples = circle(30, 0.25);
		const dropped = lassoPoints(undefined, samples, scale);
		expect(dropped[0]).toEqual(samples[0]);
		// Never closer than the spacing, and never much further.
		for (const gap of gaps(dropped, scale)) {
			expect(gap).toBeGreaterThanOrEqual(LASSO_SPACING);
			expect(gap).toBeLessThan(LASSO_SPACING + 0.5);
		}
		// About one point per spacing of the way round, not one per sample.
		const around = 2 * Math.PI * 30;
		expect(dropped.length).toBeGreaterThan(around / (LASSO_SPACING + 0.5) - 1);
		expect(dropped.length).toBeLessThanOrEqual(around / LASSO_SPACING + 1);
		expect(samples.length).toBeGreaterThan(dropped.length * 10);
	});

	it("spaces points by screen distance, so zooming in adds detail", () => {
		const samples = circle(30, 0.05);
		const near = lassoPoints(undefined, samples, [4, 4]);
		const far = lassoPoints(undefined, samples, [0.5, 0.5]);
		expect(near.length).toBeGreaterThan(far.length * 6);
		for (const gap of gaps(near, [4, 4])) expect(gap).toBeGreaterThanOrEqual(LASSO_SPACING);
		for (const gap of gaps(far, [0.5, 0.5])) expect(gap).toBeGreaterThanOrEqual(LASSO_SPACING);
	});

	it("keeps every sample of a fast drag", () => {
		const samples: Point[] = [
			[10, 10],
			[30, 10],
			[30, 30],
		];
		expect(lassoPoints([0, 10], samples, [1, 1])).toEqual(samples);
	});

	it("goes on from the polygon's last point", () => {
		expect(lassoPoints([10, 10], [[11, 10], [12, 10], [14, 10], [15, 10]], [1, 1])).toEqual([[14, 10]]);
		expect(lassoPoints([10, 10], [[11, 10]], [1, 1])).toEqual([]);
	});

	it("rasterizes a lasso drawn round a circle as that circle", () => {
		const mask = new PlaneMask(100, 100);
		mask.polygon(lassoPoints(undefined, circle(20, 0.1), [1, 1]));
		expect(mask.count).toBeGreaterThan(Math.PI * 400 * 0.97);
		expect(mask.count).toBeLessThan(Math.PI * 400 * 1.03);
	});
});
