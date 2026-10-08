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

	it("closes polygons with Enter, with Alt or Shift held too", () => {
		expect(actionFor(press("Enter"))).toBe("close-polygon");
		expect(actionFor(press("Enter", { altKey: true }))).toBe("close-polygon");
		expect(actionFor(press("Enter", { shiftKey: true }))).toBe("close-polygon");
	});

	it("keeps Backspace and Esc working while Alt is held over a polygon", () => {
		expect(actionFor(press("Backspace", { altKey: true }))).toBe("remove-point");
		expect(actionFor(press("Escape", { altKey: true }))).toBe("cancel");
		// Alt with anything else is the browser's or the system's.
		expect(actionFor(press("p", { altKey: true }))).toBeUndefined();
		expect(actionFor(press("ArrowUp", { altKey: true }))).toBeUndefined();
	});

	it("leaves form fields and browser shortcuts alone", () => {
		expect(actionFor(press("0", {}, "INPUT"))).toBeUndefined();
		expect(actionFor(press("=", { metaKey: true }))).toBeUndefined();
	});

	it("leaves Space, Enter, and arrows to focused controls", () => {
		const on = (key: string, tagName: string, type?: string) =>
			({ key, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, target: { tagName, type, isContentEditable: false } }) as unknown as KeyboardEvent;
		expect(actionFor(on("Enter", "BUTTON"))).toBeUndefined();
		expect(actionFor(on("ArrowDown", "BUTTON"))).toBeUndefined();
		expect(actionFor(on("b", "BUTTON"))).toBe("brush");
		expect(actionFor(on("ArrowUp", "INPUT", "range"))).toBeUndefined();
		expect(actionFor(on("ArrowUp", "INPUT", "checkbox"))).toBe("slice-next");
		expect(actionFor(on("b", "INPUT", "radio"))).toBe("brush");
		expect(actionFor(on("b", "INPUT", "text"))).toBeUndefined();
	});

	it("matches shortcuts by key position on other layouts", () => {
		const cyrillic = { key: "я", code: "KeyZ", ctrlKey: true, metaKey: false, altKey: false, shiftKey: false, target: null } as unknown as KeyboardEvent;
		expect(actionFor(cyrillic)).toBe("undo");
	});

	it("knows undo and redo with Ctrl or ⌘", () => {
		expect(actionFor(press("z", { ctrlKey: true }))).toBe("undo");
		expect(actionFor(press("Z", { metaKey: true, shiftKey: true }))).toBe("redo");
		expect(actionFor(press("y", { ctrlKey: true }))).toBe("redo");
		expect(actionFor(press("z"))).toBeUndefined();
	});
});
