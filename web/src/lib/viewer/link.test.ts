import { describe, expect, it } from "vitest";
import { annotatorHref, parseBox } from "./link";

describe("annotatorHref", () => {
	it("names an ROI, a box, or both", () => {
		expect(annotatorHref("p1")).toBe("/p/p1/annotate");
		expect(annotatorHref("p1", { roi: "r1" })).toBe("/p/p1/annotate?roi=r1");
		expect(annotatorHref("p1", { box: [5, 20, 30, 6, 29, 42] })).toBe("/p/p1/annotate?box=5,20,30,6,29,42");
		expect(annotatorHref("p1", { roi: "r1", box: [0, 0, 0, 1, 2, 3] })).toBe("/p/p1/annotate?roi=r1&box=0,0,0,1,2,3");
	});

	it("gives a box the annotator reads back", () => {
		const href = annotatorHref("p1", { box: [5, 20, 30, 6, 29, 42] });
		expect(parseBox(new URL(href, "https://example.org").searchParams.get("box"))).toEqual([5, 20, 30, 6, 29, 42]);
	});
});

describe("parseBox", () => {
	it("reads six whole numbers, each end past its start", () => {
		expect(parseBox("0,0,0,1,1,1")).toEqual([0, 0, 0, 1, 1, 1]);
		expect(parseBox("40,100,7,41,164,71")).toEqual([40, 100, 7, 41, 164, 71]);
	});

	it("is null for anything else", () => {
		for (const text of [null, "", "1,2,3", "0,0,0,1,1,1,1", "0,0,0,1,1,x", "-1,0,0,1,1,1", "0.5,0,0,1,1,1", " 0,0,0,1,1,1", "0,0,0,1,1,1e3", "1234567890,0,0,1234567891,1,1"]) {
			expect(parseBox(text), String(text)).toBeNull();
		}
		// Empty along an axis, or inside out.
		expect(parseBox("5,0,0,5,1,1")).toBeNull();
		expect(parseBox("0,9,0,1,3,1")).toBeNull();
	});
});
