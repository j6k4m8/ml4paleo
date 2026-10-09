import { describe, expect, it } from "vitest";
import { answerView, isNeuroglancerLink, ownWindowHref } from "./neuroglancer";

const LINK = "/neuroglancer/#!%7B%22layers%22:%5B%5D%7D";

describe("isNeuroglancerLink", () => {
	it("takes a path into this site's Neuroglancer, with or without a view", () => {
		expect(isNeuroglancerLink("/neuroglancer/")).toBe(true);
		expect(isNeuroglancerLink(LINK)).toBe(true);
		expect(isNeuroglancerLink("/neuroglancer/?x=1#!{}")).toBe(true);
	});

	it("refuses anything else", () => {
		for (const url of [
			"",
			"/",
			"/neuroglancer",
			"/neuroglancer/../api/health",
			"/neuroglancerx/",
			"//evil.example/neuroglancer/",
			"/\\evil.example/neuroglancer/",
			"https://evil.example/neuroglancer/",
			"https://site.invalid/neuroglancer/",
			"javascript:alert(1)",
			"data:text/html,<script>1</script>",
			"neuroglancer/",
			null,
			undefined,
			42,
			{ url: LINK },
		]) {
			expect(isNeuroglancerLink(url), String(url)).toBe(false);
		}
	});
});

describe("answerView", () => {
	it("shows the link the server gives", () => {
		expect(answerView({ url: LINK })).toEqual({ kind: "ready", url: LINK });
	});

	it("says there is no Neuroglancer when the server has none", () => {
		expect(answerView({ url: null })).toEqual({ kind: "unavailable" });
		expect(answerView({})).toEqual({ kind: "unavailable" });
		expect(answerView(null)).toEqual({ kind: "unavailable" });
		expect(answerView(undefined)).toEqual({ kind: "unavailable" });
	});

	it("doesn't frame a link that goes anywhere else", () => {
		for (const url of ["https://evil.example/", "//evil.example/neuroglancer/", "javascript:alert(1)", "", 7]) {
			expect(answerView({ url }).kind, String(url)).toBe("error");
		}
	});
});

describe("ownWindowHref", () => {
	const origin = "https://ml4paleo.example.org";

	it("opens the view the frame shows now", () => {
		const now = `${origin}/neuroglancer/#!%7B%22position%22:%5B1,2,3%5D%7D`;
		expect(ownWindowHref(now, origin, LINK)).toBe("/neuroglancer/#!%7B%22position%22:%5B1,2,3%5D%7D");
	});

	it("falls back to the link the tab began with", () => {
		for (const now of [
			null,
			undefined,
			"",
			"about:blank",
			"not a url",
			`${origin}/p/1/annotate`,
			`${origin}/neuroglancer/x/`,
			"https://evil.example/neuroglancer/#!{}",
			"http://ml4paleo.example.org/neuroglancer/#!{}",
		]) {
			expect(ownWindowHref(now, origin, LINK), String(now)).toBe(LINK);
		}
	});
});
