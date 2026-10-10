import { describe, expect, it } from "vitest";
import { intersection, liveKeys } from "./live";

describe("live chunk priority", () => {
	it("starts at the center, favors visible slices, and bounds huge volumes", () => {
		const keys = liveKeys([100000, 100000, 100000], [50000, 50000, 50000], [[50000, 0, 0, 50001, 100000, 100000]]);
		expect(keys).toHaveLength(128);
		expect(keys[0]).toBe("781/781/781");
		expect(keys.slice(0, 81).every((key) => key.startsWith("781/"))).toBe(true);
		expect(new Set(keys).size).toBe(keys.length);
	});
	it("prioritizes the latest brush, including far edges of a zoomed-out view", () => {
		expect(liveKeys([10000, 10000, 10000], [5000, 5000, 5000], [], [2, 3, 4])[0]).toBe("0/0/0");
	});
	it("clips edge chunks and reprioritizes navigation instead of appending a queue", () => {
		expect(liveKeys([2, 70, 65], [1, 69, 64], [])[0]).toBe("0/1/1");
		expect(liveKeys([2, 70, 65], [0, 0, 0], [])).toHaveLength(4);
		expect(liveKeys([2, 70, 65], [0, 0, 0], [])[0]).toBe("0/0/0");
	});
	it("intersects ready regions exactly, without accepting missing areas", () => {
		expect(intersection([0, 20, 30, 1, 100, 100], [0, 64, 64, 64, 128, 128])).toEqual([0, 64, 64, 1, 100, 100]);
		expect(intersection([0, 0, 0, 1, 64, 64], [0, 64, 0, 64, 128, 64])).toBeNull();
	});
});
