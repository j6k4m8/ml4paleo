import { describe, expect, it } from "vitest";
import {
	chosenClasses,
	describeWhere,
	ERASE_MODES,
	HIGHEST_CLASS,
	modeFrom,
	onlyIf,
	PAINT_MODES,
	savedPaintMode,
	summarize,
	toClassSet,
} from "./modes";

describe("onlyIf", () => {
	it("is the mode itself for anywhere, only unlabeled, and only labeled", () => {
		expect(onlyIf("any", [])).toBe("any");
		expect(onlyIf("unlabeled", [2, 3])).toBe("unlabeled");
		expect(onlyIf("labeled", [])).toBe("labeled");
	});

	it("lists the classes, once each and ascending, as the server writes them", () => {
		expect(onlyIf("classes", [3])).toBe("class:3");
		expect(onlyIf("classes", [5, 2, 3])).toBe("class:2,3,5");
		expect(onlyIf("classes", [3, 3, 2])).toBe("class:2,3");
		// Background (1) is one of the values.
		expect(onlyIf("classes", [4, 1])).toBe("class:1,4");
	});

	it("can name every class value", () => {
		const every = Array.from({ length: HIGHEST_CLASS }, (_, i) => i + 1);
		const only = onlyIf("classes", [...every].reverse());
		expect(only).toBe(`class:${every.join(",")}`);
		expect(only.split(":")[1]!.split(",")).toHaveLength(254);
	});

	it("gives each paint and erase mode a name the writer knows, or a class list", () => {
		const names = [...PAINT_MODES, ...ERASE_MODES].map((info) => info.value);
		for (const mode of names) expect(onlyIf(mode, [2])).toMatch(/^(any|unlabeled|labeled|class:2)$/);
	});
});

describe("chosenClasses", () => {
	const available = [1, 2, 3, 4];

	it("keeps the chosen classes the project has, in its order", () => {
		expect(chosenClasses([4, 2], available, 3)).toEqual([2, 4]);
		expect(chosenClasses([1], available, 3)).toEqual([1]);
	});

	it("leaves out classes the project doesn't have", () => {
		expect(chosenClasses([2, 9], available, 3)).toEqual([2]);
	});

	it("falls back to the active class when none chosen is there, or none is chosen", () => {
		expect(chosenClasses([9], available, 3)).toEqual([3]);
		expect(chosenClasses([], available, 4)).toEqual([4]);
	});

	it("falls back to the first class when the active class isn't there either", () => {
		expect(chosenClasses([], available, null)).toEqual([1]);
		expect(chosenClasses([], available, 9)).toEqual([1]);
	});

	it("has none only when there is nothing to choose from", () => {
		expect(chosenClasses([2], [], 2)).toEqual([]);
	});
});

describe("toClassSet", () => {
	it("keeps whole numbers from 1 to 254, once each and ascending", () => {
		expect(toClassSet([3, 2, 3, 1, 254])).toEqual([1, 2, 3, 254]);
	});

	it("drops what isn't a class value", () => {
		expect(toClassSet([0, -1, 255, 2.5, "3", null, NaN, 4])).toEqual([4]);
	});

	it("takes a single number as a set of one, which is how one class was saved", () => {
		expect(toClassSet(3)).toEqual([3]);
		expect(toClassSet(0)).toEqual([]);
		expect(toClassSet(300)).toEqual([]);
	});

	it("takes anything else as no classes", () => {
		for (const bad of [undefined, null, "2", {}, true]) expect(toClassSet(bad)).toEqual([]);
	});
});

describe("modeFrom and savedPaintMode", () => {
	it("keeps a mode it knows and otherwise uses the default", () => {
		expect(modeFrom("labeled", ["any", "labeled"], "any")).toBe("labeled");
		expect(modeFrom("unlabeled", ["any", "labeled"], "any")).toBe("any");
		expect(modeFrom(3, ["any"], "any")).toBe("any");
	});

	it("reads the saved mode, or from the checkbox it replaced", () => {
		expect(savedPaintMode({ paintMode: "labeled" })).toBe("labeled");
		expect(savedPaintMode({ protectLabels: true })).toBe("unlabeled");
		expect(savedPaintMode({ protectLabels: false })).toBe("any");
		expect(savedPaintMode({})).toBe("any");
	});

	it("goes by the mode over the checkbox when both are there, and never by a mode it doesn't know", () => {
		expect(savedPaintMode({ paintMode: "any", protectLabels: true })).toBe("any");
		expect(savedPaintMode({ paintMode: "nonsense", protectLabels: true })).toBe("any");
		expect(savedPaintMode({ protectLabels: "yes" })).toBe("any");
	});
});

describe("summarize", () => {
	it("says the class, or the first and how many more", () => {
		expect(summarize(["bone"])).toBe("bone");
		expect(summarize(["bone", "matrix"])).toBe("bone + 1 more");
		expect(summarize(["bone", "matrix", "tooth"])).toBe("bone + 2 more");
		expect(summarize([])).toBe("No class");
	});
});

describe("describeWhere", () => {
	it("says nothing for anywhere, and where else paint goes", () => {
		expect(describeWhere("any", [])).toBe("");
		expect(describeWhere("unlabeled", [])).toBe("only where nothing is labeled yet");
		expect(describeWhere("labeled", [])).toBe("only over labeled voxels, background too");
	});

	it("names the classes, up to three", () => {
		expect(describeWhere("classes", ["bone"])).toBe("only over bone");
		expect(describeWhere("classes", ["bone", "matrix"])).toBe("only over bone and matrix");
		expect(describeWhere("classes", ["bone", "matrix", "tooth"])).toBe("only over bone, matrix, and tooth");
		expect(describeWhere("classes", ["a", "b", "c", "d"])).toBe("only over 4 classes");
	});
});

describe("the modes people choose from", () => {
	it("start with the plain one, and each says what it does", () => {
		expect(PAINT_MODES.map((info) => info.value)).toEqual(["any", "unlabeled", "labeled", "classes"]);
		expect(ERASE_MODES.map((info) => info.value)).toEqual(["any", "classes"]);
		for (const info of [...PAINT_MODES, ...ERASE_MODES]) {
			expect(info.label).not.toBe("");
			expect(info.title.length).toBeGreaterThan(10);
		}
		expect(PAINT_MODES.find((info) => info.value === "unlabeled")?.title).toBe("Paint only where nothing is labeled yet");
	});
});
