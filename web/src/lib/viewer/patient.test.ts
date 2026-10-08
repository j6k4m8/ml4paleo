import { describe, expect, it } from "vitest";
import { retryDelay } from "./patient";

describe("retryDelay", () => {
	it("waits what the server says, and up to half as long again at random", () => {
		expect(retryDelay("1", 0)).toBe(1000);
		expect(retryDelay("1", 1)).toBe(1500);
		expect(retryDelay("2", 0.5)).toBe(2500);
	});

	it("never waits less than half a second or more than half a minute", () => {
		expect(retryDelay("0", 0)).toBe(500);
		expect(retryDelay("3600", 0)).toBe(30_000);
	});

	it("waits two seconds when the server doesn't say", () => {
		for (const header of [null, "", "soon"]) expect(retryDelay(header, 0)).toBe(2000);
	});

	it("reads a date too", () => {
		const now = Date.parse("Wed, 21 Oct 2026 07:28:00 GMT");
		expect(retryDelay("Wed, 21 Oct 2026 07:28:04 GMT", 0, now)).toBe(4000);
		// One already past waits the least.
		expect(retryDelay("Wed, 21 Oct 2026 07:27:00 GMT", 0, now)).toBe(500);
	});
});
