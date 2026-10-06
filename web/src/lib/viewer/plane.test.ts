import { describe, expect, it } from "vitest";
import { levelFactors } from "./image";
import { planeSlice } from "./plane";

describe("planeSlice", () => {
	it("takes one z slice of a chunk as floats", () => {
		// 2 × 2 × 3 (z, y, x), values 0..11
		const data = new Uint16Array(Array.from({ length: 12 }, (_, i) => i));
		const slice = planeSlice({ data, shape: [2, 2, 3] }, 1);
		expect(slice.width).toBe(3);
		expect(slice.height).toBe(2);
		expect(Array.from(slice.data)).toEqual([6, 7, 8, 9, 10, 11]);
	});

	it("keeps 16-bit and negative values", () => {
		const data = new Int16Array([-30000, 30000]);
		expect(Array.from(planeSlice({ data, shape: [1, 1, 2] }, 0).data)).toEqual([-30000, 30000]);
	});
});

describe("levelFactors", () => {
	it("gives each level's downsampling relative to level 0", () => {
		const scale = (s: number[]) => ({ path: "", coordinateTransformations: [{ type: "scale", scale: s }] });
		const factors = levelFactors({
			datasets: [scale([1, 4, 0.5, 0.5]), scale([1, 4, 1, 1]), scale([1, 8, 2, 2])],
		});
		expect(factors).toEqual([
			[1, 1, 1],
			[1, 2, 2],
			[2, 4, 4],
		]);
	});
});
