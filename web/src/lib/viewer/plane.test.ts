import { describe, expect, it } from "vitest";
import { levelFactors } from "./image";
import { paletteBytes, planeSlice } from "./plane";
import { PLANES } from "./tiles";

describe("planeSlice", () => {
	// 2 × 2 × 3 (z, y, x): value = 100 z + 10 y + x
	const values: number[] = [];
	for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 3; x++) values.push(100 * z + 10 * y + x);
	const chunk = { data: new Uint16Array(values), shape: [2, 2, 3] };

	it("cuts XY slices with x across and y down", () => {
		const slice = planeSlice(chunk, PLANES.xy, 1, Float32Array);
		expect([slice.width, slice.height]).toEqual([3, 2]);
		expect(Array.from(slice.data)).toEqual([100, 101, 102, 110, 111, 112]);
	});

	it("cuts XZ slices with x across and z down", () => {
		const slice = planeSlice(chunk, PLANES.xz, 1, Float32Array);
		expect([slice.width, slice.height]).toEqual([3, 2]);
		expect(Array.from(slice.data)).toEqual([10, 11, 12, 110, 111, 112]);
	});

	it("cuts YZ slices with z across and y down", () => {
		const slice = planeSlice(chunk, PLANES.yz, 2, Uint8Array);
		expect([slice.width, slice.height]).toEqual([2, 2]);
		expect(Array.from(slice.data)).toEqual([2, 102, 12, 112].map((v) => v & 255));
	});

	it("keeps 16-bit and negative values", () => {
		const data = new Int16Array([-30000, 30000]);
		expect(Array.from(planeSlice({ data, shape: [1, 1, 2] }, PLANES.xy, 0, Float32Array).data)).toEqual([
			-30000, 30000,
		]);
	});
});

describe("paletteBytes", () => {
	it("colors label values and leaves the rest clear", () => {
		const bytes = paletteBytes(new Map([[2, "#ff8000"], [3, "red"], [0, "#ffffff"]]));
		expect(Array.from(bytes.slice(8, 12))).toEqual([255, 128, 0, 255]);
		expect(Array.from(bytes.slice(0, 4))).toEqual([0, 0, 0, 0]);
		expect(Array.from(bytes.slice(12, 16))).toEqual([0, 0, 0, 0]);
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
