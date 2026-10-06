import { describe, expect, it } from "vitest";
import { thumbnailLevel, thumbnailPlane } from "./thumbnail";
import type { Level } from "./viewer/tiles";

const levels: Level[] = [
	{ index: 0, path: "0", shape: [100, 1000, 1000], scale: [1, 1, 1] },
	{ index: 1, path: "1", shape: [100, 500, 500], scale: [1, 2, 2] },
	{ index: 2, path: "2", shape: [50, 250, 250], scale: [2, 4, 4] },
];

describe("thumbnails", () => {
	it("show slice ROIs on their own plane", () => {
		expect(thumbnailPlane([5, 0, 0, 6, 64, 64]).name).toBe("xy");
		expect(thumbnailPlane([0, 9, 0, 64, 10, 64]).name).toBe("xz");
		expect(thumbnailPlane([0, 0, 3, 64, 64, 4]).name).toBe("yz");
		expect(thumbnailPlane([0, 0, 0, 64, 64, 64]).name).toBe("xy");
	});

	it("use the coarsest level still about thumbnail size", () => {
		const level = (box: [number, number, number, number, number, number]) =>
			thumbnailLevel(levels, box, thumbnailPlane(box)).index;
		expect(level([0, 0, 0, 1, 64, 64])).toBe(0);
		expect(level([0, 0, 0, 1, 200, 200])).toBe(1);
		expect(level([0, 0, 0, 1, 400, 400])).toBe(2);
	});
});
