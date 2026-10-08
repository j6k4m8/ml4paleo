import { describe, expect, it } from "vitest";
import { BACKGROUND_CLASS, BACKGROUND_VALUE, withBackground } from "./background";

describe("withBackground", () => {
	it("puts background first, ahead of the project's classes, as value 1", () => {
		const bone = { value: 2, name: "bone", color: "#ffffff" };
		expect(withBackground([bone])).toEqual([BACKGROUND_CLASS, bone]);
		expect(withBackground([])).toEqual([BACKGROUND_CLASS]);
		expect(BACKGROUND_CLASS.value).toBe(BACKGROUND_VALUE);
		expect(BACKGROUND_VALUE).toBe(1);
	});
});
