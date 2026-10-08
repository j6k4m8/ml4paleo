import { describe, expect, it } from "vitest";
import { PLANES } from "../viewer/tiles";
import { applyLocally, decodeDelta, splitIntoDeltas } from "./deltas";
import {
	apart,
	CLOSE_PIXELS,
	closesAt,
	closingMode,
	LASSO_SPACING,
	lassoPoints,
	type Point,
	type Polygon,
	polygonEdit,
	type Scale,
} from "./polygon";

const square = (u0: number, v0: number, u1: number, v1: number): Point[] => [
	[u0, v0],
	[u1, v0],
	[u1, v1],
	[u0, v1],
];

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
		const polygon: Polygon = { plane: "xy", slice: 0, points: lassoPoints(undefined, circle(20, 0.1), [1, 1]) };
		const edit = polygonEdit(polygon, "add", 2, false, [100, 100]);
		expect(edit?.mask.count).toBeGreaterThan(Math.PI * 400 * 0.97);
		expect(edit?.mask.count).toBeLessThan(Math.PI * 400 * 1.03);
	});
});

describe("closingMode", () => {
	const none = { altKey: false, shiftKey: false };

	it("follows the mode with no keys held", () => {
		expect(closingMode("add", none)).toBe("add");
		expect(closingMode("subtract", none)).toBe("subtract");
	});

	it("cuts out with Alt and fills with Shift, whatever the mode", () => {
		expect(closingMode("add", { ...none, altKey: true })).toBe("subtract");
		expect(closingMode("subtract", { ...none, altKey: true })).toBe("subtract");
		expect(closingMode("subtract", { ...none, shiftKey: true })).toBe("add");
		expect(closingMode("add", { ...none, shiftKey: true })).toBe("add");
		expect(closingMode("add", { altKey: true, shiftKey: true })).toBe("subtract");
	});
});

describe("polygonEdit", () => {
	const SIZE = 32;

	/** One XY slice of label values, SIZE × SIZE, as a chunk at the image's corner. */
	function slice(): Uint8Array {
		return new Uint8Array(SIZE * SIZE);
	}

	/** Apply an edit to the slice, as the label overlay does before the server answers. */
	function apply(labels: Uint8Array, polygon: Polygon, mode: "add" | "subtract", value: number, protect = false) {
		const edit = polygonEdit(polygon, mode, value, protect, [SIZE, SIZE]);
		if (!edit) throw new Error("The polygon covers nothing");
		const volume = edit.mask.toVolume(PLANES.xy, polygon.slice);
		for (const delta of splitIntoDeltas(volume.mask, volume.shape, volume.origin, { value: edit.value, onlyIf: edit.onlyIf })) {
			const { mask, written } = decodeDelta(delta);
			applyLocally(labels, [1, SIZE, SIZE], delta.box, mask, written, delta.only_if);
		}
		return edit;
	}

	const at = (labels: Uint8Array, u: number, v: number) => labels[v * SIZE + u];

	it("cuts a hole out of the active class, leaving other classes inside", () => {
		const labels = slice();
		apply(labels, { plane: "xy", slice: 0, points: square(4, 4, 28, 28) }, "add", 2);
		// Someone labeled a voxel of another class where the hole goes.
		labels[16 * SIZE + 16] = 3;
		const edit = apply(labels, { plane: "xy", slice: 0, points: square(10, 10, 22, 22) }, "subtract", 2);
		expect(edit.value).toBe(0);
		expect(edit.onlyIf).toBe("class:2");
		expect(edit.tool.name).toBe("polygon-erase");
		// The ring stays, the hole is unlabeled, and the other class is kept.
		expect(at(labels, 4, 4)).toBe(2);
		expect(at(labels, 27, 27)).toBe(2);
		expect(at(labels, 9, 16)).toBe(2);
		expect(at(labels, 22, 16)).toBe(2);
		expect(at(labels, 10, 10)).toBe(0);
		expect(at(labels, 21, 21)).toBe(0);
		expect(at(labels, 16, 16)).toBe(3);
		expect(at(labels, 3, 3)).toBe(0);
		expect(at(labels, 28, 28)).toBe(0);
		const counts = [0, 2, 3].map((value) => labels.filter((l) => l === value).length);
		expect(counts).toEqual([SIZE * SIZE - (24 * 24 - 12 * 12) - 1, 24 * 24 - 12 * 12, 1]);
	});

	it("cuts out only the active class where two classes meet", () => {
		const labels = slice();
		apply(labels, { plane: "xy", slice: 0, points: square(0, 0, 16, 32) }, "add", 2);
		apply(labels, { plane: "xy", slice: 0, points: square(16, 0, 32, 32) }, "add", 4);
		// A cutout across both halves takes out only class 2.
		apply(labels, { plane: "xy", slice: 0, points: square(8, 8, 24, 24) }, "subtract", 2);
		expect(at(labels, 12, 12)).toBe(0);
		expect(at(labels, 20, 12)).toBe(4);
		expect(at(labels, 4, 12)).toBe(2);
	});

	it("fills with the class, only into unlabeled voxels when asked", () => {
		const labels = slice();
		labels[5 * SIZE + 5] = 3;
		const edit = apply(labels, { plane: "xy", slice: 0, points: square(2, 2, 8, 8) }, "add", 2, true);
		expect(edit).toMatchObject({ value: 2, onlyIf: "unlabeled", tool: { name: "polygon", plane: "xy", slice: 0 } });
		expect(at(labels, 5, 5)).toBe(3);
		expect(at(labels, 4, 4)).toBe(2);
		expect(polygonEdit({ plane: "xy", slice: 0, points: square(2, 2, 8, 8) }, "add", 2, false, [SIZE, SIZE])?.onlyIf).toBe("any");
	});

	it("records the outline, rounded, unless it's very long", () => {
		const edit = polygonEdit({ plane: "yz", slice: 3, points: [[1.234, 2], [9.87, 2], [5, 7.06]] }, "add", 2, false, [SIZE, SIZE]);
		expect(edit?.tool).toEqual({ name: "polygon", plane: "yz", slice: 3, points: [[1.2, 2], [9.9, 2], [5, 7.1]] });
		const long = Array.from({ length: 600 }, (_, i): Point => {
			const angle = (2 * Math.PI * i) / 600;
			return [16 + 10 * Math.cos(angle), 16 + 10 * Math.sin(angle)];
		});
		expect(polygonEdit({ plane: "xy", slice: 0, points: long }, "add", 2, false, [SIZE, SIZE])?.tool.points).toBeUndefined();
	});

	it("makes nothing of a polygon with no voxel centers inside", () => {
		expect(polygonEdit({ plane: "xy", slice: 0, points: [[1, 1], [5, 1]] }, "add", 2, false, [SIZE, SIZE])).toBeNull();
		expect(polygonEdit({ plane: "xy", slice: 0, points: [[1, 1], [5, 1], [9, 1]] }, "subtract", 2, false, [SIZE, SIZE])).toBeNull();
	});
});
