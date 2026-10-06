import { describe, expect, it } from "vitest";
import { safeNext } from "./navigation";

describe("safeNext", () => {
	const origin = "https://m4p.example";

	it("keeps paths on this site", () => {
		expect(safeNext("/p/1/annotate?roi=2#x", origin)).toBe("/p/1/annotate?roi=2#x");
		expect(safeNext(null, origin)).toBe("/projects");
	});

	it("refuses other sites however they're spelled", () => {
		for (const target of ["//evil.example", "/\\evil.example", "/\t/evil.example", "https://evil.example/", "javascript:alert(1)"]) {
			expect(safeNext(target, origin)).toBe("/projects");
		}
	});
});
