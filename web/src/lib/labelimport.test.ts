import { describe, expect, it } from "vitest";
import { CLASS_COLORS, defaultTargets, labelFile, newClasses, nextColor, startBody, type Target } from "./labelimport";

const bone = { value: 2, name: "Bone", color: CLASS_COLORS[0]! };

describe("defaultTargets", () => {
	it("leaves 0 unlabeled and makes new classes named after the other values", () => {
		const targets = defaultTargets([0, 1, 7], []);
		expect(targets.get(0)).toEqual({ kind: "skip" });
		expect(targets.get(1)).toEqual({ kind: "new", name: "1", color: CLASS_COLORS[0] });
		expect(targets.get(7)).toEqual({ kind: "new", name: "7", color: CLASS_COLORS[1] });
	});

	it("uses a class named after a value, and the only class for the only value", () => {
		const seven = { value: 3, name: "7", color: "#000000" };
		expect(defaultTargets([0, 1, 7], [bone, seven]).get(7)).toEqual({ kind: "label", label: 3 });
		expect(defaultTargets([0, 255], [bone]).get(255)).toEqual({ kind: "label", label: 2 });
		// With more than one value, which is which isn't clear.
		expect(defaultTargets([1, 2], [bone]).get(1)).toMatchObject({ kind: "new" });
	});

	it("gives new classes colors no class has", () => {
		const targets = newClasses([1], [bone]);
		expect(targets.get(1)).toMatchObject({ color: CLASS_COLORS[1] });
		expect(nextColor(CLASS_COLORS)).toBe(CLASS_COLORS[0]);
	});
});

describe("startBody", () => {
	it("sends what each value becomes, leaving unlabeled ones out", () => {
		const targets = new Map<number, Target>([
			[0, { kind: "skip" }],
			[1, { kind: "label", label: 1 }],
			[7, { kind: "new", name: " Shell ", color: "#46a758" }],
		]);
		expect(startBody(targets, true)).toEqual({
			mapping: [
				{ value: 1, label: 1 },
				{ value: 7, new_class: { name: "Shell", color: "#46a758" } },
			],
			overwrite: true,
		});
	});

	it("says why it can't start", () => {
		expect(startBody(new Map([[0, { kind: "skip" }]]), false)).toBe("Choose what at least one value becomes.");
		expect(startBody(new Map([[1, { kind: "new", name: " ", color: "#000000" }]]), false)).toBe(
			"Give each new class a name.",
		);
	});
});

describe("labelFile", () => {
	it("takes TIFF, PNG, and zip files", () => {
		expect(["a.tif", "b.TIFF", "c.png", "d.zip"].every(labelFile)).toBe(true);
		expect(["e.nrrd", "f.tif.gz", "g"].some(labelFile)).toBe(false);
	});
});
