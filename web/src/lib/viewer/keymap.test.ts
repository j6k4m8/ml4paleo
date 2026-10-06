import { describe, expect, it } from "vitest";
import { actionFor, KEYMAP } from "./keymap";

function press(key: string, init: Partial<KeyboardEventInit> = {}, tagName = "CANVAS") {
	return { key, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, ...init, target: { tagName, isContentEditable: false } } as unknown as KeyboardEvent;
}

describe("keymap", () => {
	it("binds each key once", () => {
		const keys = KEYMAP.flatMap((binding) => binding.keys);
		expect(new Set(keys).size).toBe(keys.length);
	});

	it("maps presses to actions", () => {
		expect(actionFor(press("ArrowUp"))).toBe("slice-next");
		expect(actionFor(press(">", { shiftKey: true }))).toBe("slice-next");
		expect(actionFor(press("L", { shiftKey: true }))).toBe("layout");
		expect(actionFor(press("?", { shiftKey: true }))).toBe("help");
	});

	it("leaves form fields and shortcuts alone", () => {
		expect(actionFor(press("0", {}, "INPUT"))).toBeUndefined();
		expect(actionFor(press("=", { metaKey: true }))).toBeUndefined();
	});
});
