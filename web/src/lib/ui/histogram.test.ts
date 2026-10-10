import { describe, expect, it } from "vitest";
import { displayLevel, editLevel, handleAt } from "./histogram";

describe("handleAt", () => {
	it("grabs the nearer end between them, black on a tie", () => {
		expect(handleAt(20, [10, 100])).toBe(0);
		expect(handleAt(90, [10, 100])).toBe(1);
		expect(handleAt(55, [10, 100])).toBe(0);
	});

	it("grabs the end on the side pressed outside them", () => {
		expect(handleAt(0, [10, 100])).toBe(0);
		expect(handleAt(200, [10, 100])).toBe(1);
	});

	it("pulls ends that meet apart toward the press", () => {
		expect(handleAt(150, [100, 100])).toBe(1);
		expect(handleAt(50, [100, 100])).toBe(0);
	});
});

describe("whole-number level controls", () => {
	it("rounds floating-point tails without changing the actual contrast window", () => {
		const window: [number, number] = [19, 2198.4049999999997];
		expect(window.map(displayLevel)).toEqual([19, 2198]);
		expect(window[1]).toBe(2198.4049999999997);
		expect(displayLevel(-12.7)).toBe(-13);
	});
	it("commits integer edits and preserves the other endpoint's precision", () => {
		expect(editLevel([0.0123, 0.9876], 1, 100.8)).toEqual([0.0123, 101]);
		expect(editLevel([0.0123, 0.9876], 0, -8.3)).toEqual([-8, 0.9876]);
	});
	it.each([undefined, NaN, Infinity, -Infinity])("ignores an empty or invalid number: %s", (value) => {
		expect(editLevel([19, 2198.405], 1, value)).toEqual([19, 2198.405]);
	});
});
