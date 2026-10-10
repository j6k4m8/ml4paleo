import { afterEach, describe, expect, it, vi } from "vitest";
import { classOpacity, displayedValues, readClassStyles } from "./class-display";
import { ClassDisplay } from "./class-display.svelte";
import { unlabeledOnly } from "../labels/accept";

afterEach(() => vi.unstubAllGlobals());

describe("class display preferences", () => {
	it("defaults to visible and clamps opacity to a safe integer percent", () => {
		expect(classOpacity(2, {})).toBe(1);
		expect(classOpacity(2, { 2: { visible: false, opacity: 1 } })).toBe(0);
		expect(classOpacity(2, { 2: { opacity: 0.237 } })).toBe(0.24);
		expect(classOpacity(2, { 2: { opacity: NaN } })).toBe(1);
		expect(classOpacity(2, { 2: { opacity: -1 } })).toBe(0);
		expect(classOpacity(2, { 2: { opacity: 2 } })).toBe(1);
	});

	it("only restores valid class settings and only permits a local Background color", () => {
		expect(readClassStyles(null)).toEqual({});
		expect(readClassStyles([])).toEqual({});
		expect(readClassStyles({
			0: { opacity: 0 }, 255: { visible: true }, "-1": {}, x: {},
			1: { color: "#ABCDEF", opacity: 0.5 }, 2: { color: "#123456", visible: false, opacity: Infinity },
		})).toEqual({ 1: { color: "#abcdef", opacity: 0.5 }, 2: { visible: false } });
	});

	it("filters decisions without mutating predictions or treating hidden saved paint as unlabeled", () => {
		const predicted = Uint8Array.from([0, 1, 2, 3, 4, 4, 255]);
		const saved = Uint8Array.from([0, 0, 0, 0, 2, 0, 0]);
		const filtered = displayedValues(unlabeledOnly(predicted, saved), {
			1: { visible: false }, 2: { visible: false }, 3: { opacity: 0 }, 4: { opacity: 0.25 },
		});
		expect([...filtered]).toEqual([0, 0, 0, 0, 0, 4, 0]);
		expect([...predicted]).toEqual([0, 1, 2, 3, 4, 4, 255]);
		expect([...saved]).toEqual([0, 0, 0, 0, 2, 0, 0]);
	});

	it("persists per project and user, retaining opacity while toggling visibility", () => {
		const data = new Map<string, string>();
		vi.stubGlobal("localStorage", { getItem: (key: string) => data.get(key), setItem: (key: string, value: string) => data.set(key, value) });
		const display = new ClassDisplay("project", "user");
		display.set(2, { opacity: 0.4 });
		display.set(2, { visible: false });
		display.set(1, { color: "#123456" });
		expect(new ClassDisplay("project", "user").styles).toEqual(display.styles);
		expect(new ClassDisplay("other", "user").styles).toEqual({});
		expect(new ClassDisplay("project", "other").styles).toEqual({});
		expect(display.reveal(2)).toBe(true);
		expect(classOpacity(2, display.styles)).toBe(0.4);
		display.set(2, { opacity: 0 });
		expect(display.reveal(2)).toBe(true);
		expect(classOpacity(2, display.styles)).toBe(1);
		expect(display.reveal(2)).toBe(false);
		expect(display.backgroundColor).toBe("#123456");
	});

	it("keeps working when browser storage is corrupt or unavailable", () => {
		vi.stubGlobal("localStorage", { getItem: () => "{", setItem: () => { throw new Error("storage disabled"); } });
		const display = new ClassDisplay("project", "user");
		expect(display.styles).toEqual({});
		display.set(2, { opacity: 0.3 });
		expect(classOpacity(2, display.styles)).toBe(0.3);
	});
});
