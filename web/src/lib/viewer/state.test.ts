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
	it("uses Accept's selected shape without needing an active paint class", () => {
		const viewer = viewerWith({});
		viewer.tool = "accept";
		expect(viewer.activeClass).toBeNull();
		expect([viewer.drawingBrush, viewer.drawingPolygon]).toEqual([true, false]);
		viewer.acceptShape = "polygon";
		expect([viewer.drawingBrush, viewer.drawingPolygon]).toEqual([false, true]);
		viewer.tool = "brush";
		expect([viewer.drawingBrush, viewer.drawingPolygon]).toEqual([true, false]);
		viewer.tool = "polygon";
		expect([viewer.drawingBrush, viewer.drawingPolygon]).toEqual([false, true]);
	});

	it("starts with predictions hidden even when a previous visit showed them", () => {
		expect(viewerWith({ showPrediction: true }).showPrediction).toBe(false);
	});

	it("accepts every foreground class by default, independent of paint and erase filters", () => {
		const viewer = viewerWith({ paintMode: "classes", paintClasses: [2], eraseMode: "classes", eraseClasses: [3] });
		expect(viewer.acceptMode).toBe("any");
		expect(viewer.acceptClasses).toEqual([]);
		expect(viewer.acceptValues([1, 2, 3, 4])).toEqual([2, 3, 4]);
		viewer.acceptClasses = [3];
		// A remembered subset only takes effect in Only classes.
		expect(viewer.acceptValues([1, 2, 3, 4])).toEqual([2, 3, 4]);
	});

	it.each(["brush", "polygon"] as const)("uses the selected predicted classes for an Accept %s", (shape) => {
		const viewer = viewerWith({ acceptMode: "classes", acceptClasses: [4, 2] });
		viewer.tool = "accept";
		viewer.acceptShape = shape;
		viewer.activeClass = 3;
		expect(viewer.acceptValues([1, 2, 3, 4])).toEqual([2, 4]);
		const frozen = new Set(viewer.acceptValues([1, 2, 3, 4]));
		viewer.acceptClasses = [3];
		expect([...frozen]).toEqual([2, 4]);
		expect(viewer.acceptValues([1, 2, 3, 4])).toEqual([3]);
	});

	it("falls back to an available foreground class, never Background, for Accept", () => {
		const viewer = viewerWith({ acceptMode: "classes", acceptClasses: [8, 9] });
		viewer.activeClass = 3;
		expect(viewer.classesFor("accept", [1, 2, 3])).toEqual([3]);
		viewer.activeClass = 1;
		expect(viewer.acceptValues([1, 2, 3])).toEqual([2]);
		expect(viewer.acceptValues([1])).toEqual([]);
		viewer.acceptClasses = [3, 9];
		expect(viewer.acceptValues([1, 2, 3])).toEqual([3]);
	});

	it("sanitizes saved Accept filters and remembers them across visits", () => {
		const viewer = viewerWith({ acceptMode: "classes", acceptClasses: [4, 2, 2, 1, 0, 255, 2.5, "3", null] });
		expect(viewer.acceptClasses).toEqual([2, 4]);
		viewer.savePreferences();
		const restored = new ViewerState([10, 10, 10], [1, 1, 1]);
		expect([restored.acceptMode, restored.acceptClasses]).toEqual(["classes", [2, 4]]);
		expect(viewerWith({ acceptMode: "labeled", acceptClasses: "2" }).acceptMode).toBe("any");
		expect(viewerWith({ acceptClasses: [1, 255, null] }).acceptClasses).toEqual([]);
	});
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

	it("paints anywhere and erases anything until told otherwise", () => {
		const viewer = viewerWith({});
		expect([viewer.paintMode, viewer.eraseMode]).toEqual(["any", "any"]);
		expect([viewer.paintClasses, viewer.eraseClasses]).toEqual([[], []]);
		expect(viewer.paintCondition([1, 2, 3])).toBe("any");
		expect(viewer.eraseCondition([1, 2, 3])).toBe("any");
	});

	it("keeps a saved paint mode and the classes it goes by, and the eraser's", () => {
		const viewer = viewerWith({ paintMode: "classes", paintClasses: [4, 2], eraseMode: "classes", eraseClasses: [1, 3] });
		expect([viewer.paintMode, viewer.paintClasses]).toEqual(["classes", [2, 4]]);
		expect([viewer.eraseMode, viewer.eraseClasses]).toEqual(["classes", [1, 3]]);
		expect(viewerWith({ paintMode: "labeled" }).paintMode).toBe("labeled");
		expect(viewerWith({ paintMode: "unlabeled" }).paintMode).toBe("unlabeled");
		expect(viewerWith({ eraseMode: "predictions" }).eraseMode).toBe("predictions");
	});

	it("migrates the old checkbox for painting only unlabeled voxels to that mode", () => {
		expect(viewerWith({ protectLabels: true }).paintMode).toBe("unlabeled");
		expect(viewerWith({ protectLabels: false }).paintMode).toBe("any");
		// A mode saved since wins over the checkbox.
		expect(viewerWith({ protectLabels: true, paintMode: "labeled" }).paintMode).toBe("labeled");
		expect(viewerWith({ protectLabels: true, paintMode: "any" }).paintMode).toBe("any");
	});

	it("saves the mode it migrated to, and no longer the checkbox", () => {
		const saved = storage({ protectLabels: true });
		vi.stubGlobal("localStorage", saved);
		const viewer = new ViewerState([10, 10, 10], [1, 1, 1]);
		viewer.savePreferences();
		const written = JSON.parse(saved.items.get("m4p.viewer") ?? "{}");
		expect(written).toMatchObject({ paintMode: "unlabeled", paintClasses: [], eraseMode: "any", eraseClasses: [] });
		expect(written).not.toHaveProperty("protectLabels");
		// And the next visit reads it back.
		expect(new ViewerState([10, 10, 10], [1, 1, 1]).paintMode).toBe("unlabeled");
	});

	it("migrates a single saved class to a set of one", () => {
		const viewer = viewerWith({ paintMode: "classes", paintClass: 3, eraseMode: "classes", eraseClass: 2 });
		expect(viewer.paintClasses).toEqual([3]);
		expect(viewer.eraseClasses).toEqual([2]);
		expect(viewer.paintCondition([1, 2, 3])).toBe("class:3");
		expect(viewer.eraseCondition([1, 2, 3])).toBe("class:2");
		// A set saved since wins over a single class.
		expect(viewerWith({ paintClasses: [2, 5], paintClass: 3 }).paintClasses).toEqual([2, 5]);
		// And a number saved as the set is a set of one too.
		expect(viewerWith({ paintClasses: 4 }).paintClasses).toEqual([4]);
	});

	it("ignores saved modes and classes it can't use", () => {
		const viewer = viewerWith({ paintMode: "everywhere", eraseMode: "labeled", paintClasses: ["a", 0, 300, null], eraseClasses: "2" });
		expect([viewer.paintMode, viewer.eraseMode]).toEqual(["any", "any"]);
		expect([viewer.paintClasses, viewer.eraseClasses]).toEqual([[], []]);
	});

	it("says what to send for each mode, going by the classes this project has", () => {
		const viewer = viewerWith({});
		viewer.activeClass = 3;
		viewer.paintMode = "unlabeled";
		expect(viewer.paintCondition([1, 2, 3])).toBe("unlabeled");
		viewer.paintMode = "labeled";
		expect(viewer.paintCondition([1, 2, 3])).toBe("labeled");
		viewer.paintMode = "classes";
		viewer.paintClasses = [3, 1];
		expect(viewer.paintCondition([1, 2, 3])).toBe("class:1,3");
		// A class this project doesn't have is left out, and with none left the active class is used.
		viewer.paintClasses = [2, 9];
		expect(viewer.paintCondition([1, 2, 3])).toBe("class:2");
		viewer.paintClasses = [8, 9];
		expect(viewer.paintCondition([1, 2, 3])).toBe("class:3");
		viewer.eraseMode = "classes";
		viewer.eraseClasses = [2];
		expect(viewer.eraseCondition([1, 2, 3])).toBe("class:2");
		// The eraser's classes are its own.
		expect(viewer.paintCondition([1, 2, 3])).toBe("class:3");
		viewer.eraseMode = "predictions";
		expect(viewer.eraseCondition([1, 2, 3])).toBe("unlabeled");
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
