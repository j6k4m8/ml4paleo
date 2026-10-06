import { describe, expect, it } from "vitest";
import { handleAt } from "./histogram";

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
