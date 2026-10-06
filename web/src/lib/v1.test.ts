import { describe, expect, it } from "vitest";
import { parseJobId, rememberedJobs } from "./v1";

describe("parseJobId", () => {
	it("takes an id or a v1 link", () => {
		expect(parseJobId(" ab12cd ")).toBe("AB12CD");
		expect(parseJobId("https://ml4paleo.example/job/AB12CD")).toBe("AB12CD");
		expect(parseJobId("https://ml4paleo.example/job/ab12cd/annotate?x=1")).toBe("AB12CD");
	});

	it("refuses anything else", () => {
		for (const text of ["", "AB12C", "AB12CDE", "XY12CD", "https://ml4paleo.example/p/AB12CD"]) {
			expect(parseJobId(text)).toBeNull();
		}
	});
});

describe("rememberedJobs", () => {
	const storage = (value: string | null) => ({ getItem: () => value });

	it("reads v1's list once per job", () => {
		const saved = JSON.stringify([
			{ id: "AB12CD", name: "Burrow" },
			{ id: "ab12cd", name: "again" },
			{ id: "nope", name: "bad" },
			{ id: "00FF00" },
			"junk",
		]);
		expect(rememberedJobs(storage(saved))).toEqual([
			{ id: "AB12CD", name: "Burrow" },
			{ id: "00FF00", name: "" },
		]);
	});

	it("copes with nothing saved or something else saved", () => {
		expect(rememberedJobs(null)).toEqual([]);
		expect(rememberedJobs(storage(null))).toEqual([]);
		expect(rememberedJobs(storage("{not json"))).toEqual([]);
		expect(rememberedJobs(storage('{"id": "AB12CD"}'))).toEqual([]);
	});
});
