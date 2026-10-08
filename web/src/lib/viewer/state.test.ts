import { afterEach, describe, expect, it, vi } from "vitest";
import { ViewerState } from "./state.svelte";

/** A browser's storage holding the viewer preferences `saved`. */
function storage(saved: Record<string, unknown>) {
	const items = new Map([["m4p.viewer", JSON.stringify(saved)]]);
	return {
		items,
		getItem: (key: string) => items.get(key) ?? null,
		setItem: (key: string, value: string) => void items.set(key, value),
	};
}

const viewerWith = (saved: Record<string, unknown>) => {
	vi.stubGlobal("localStorage", storage(saved));
	return new ViewerState([10, 10, 10], [1, 1, 1]);
};

describe("ViewerState", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("shows labels on every visit, even if they were hidden on the last", () => {
		const viewer = viewerWith({ showLabels: false, opacity: 0.7, layout: "xy" });
		expect(viewer.showLabels).toBe(true);
		expect(viewer.opacity).toBe(0.7);
		expect(viewer.layout).toBe("xy");
	});

	it("doesn't bring back labels too faint to see", () => {
		expect(viewerWith({ opacity: 0 }).opacity).toBe(0.5);
		expect(viewerWith({ opacity: 0.05 }).opacity).toBe(0.5);
	});

	it("shows hidden or faint labels again for an edit, and leaves visible ones alone", () => {
		const viewer = viewerWith({});
		viewer.showLabels = false;
		viewer.opacity = 0.05;
		viewer.revealLabels();
		expect(viewer.showLabels).toBe(true);
		expect(viewer.opacity).toBe(0.5);
		viewer.opacity = 0.3;
		viewer.revealLabels();
		expect(viewer.opacity).toBe(0.3);
	});

	it("doesn't save whether labels show", () => {
		const saved = storage({});
		vi.stubGlobal("localStorage", saved);
		const viewer = new ViewerState([10, 10, 10], [1, 1, 1]);
		viewer.showLabels = false;
		viewer.savePreferences();
		expect(JSON.parse(saved.items.get("m4p.viewer") ?? "{}")).not.toHaveProperty("showLabels");
	});
});
