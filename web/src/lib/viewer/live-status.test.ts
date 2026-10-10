import { describe, expect, it } from "vitest";
import { render } from "svelte/server";
import LiveStatus from "./LiveStatus.svelte";

describe("compact suggestion status", () => {
	it("shows short labels, queued edits, and accessible numeric progress", () => {
		const { body } = render(LiveStatus, { props: {
			learning: "learning", prediction: "predicting", queuedEdits: true, progress: { ready: 24, total: 56 },
		} });
		expect(body).toContain("Learning");
		expect(body).toContain("Predicting");
		expect(body).toContain("Edits queued");
		expect(body).toContain("24/56");
		expect(body).toContain('aria-valuenow="24"');
		expect(body).toContain('aria-valuemax="56"');
		expect(body).not.toContain("Learning from your saved labels");
	});
	it.each([{ paused: true, label: "Paused" }, { held: true, label: "On hold" }])("does not imply active work while $label", ({ label, ...state }) => {
		const { body } = render(LiveStatus, { props: {
			learning: "learning", prediction: "predicting", progress: { ready: 0, total: 56 }, ...state,
		} });
		expect(body).toContain(label);
		expect(body).not.toContain("animate-spin");
		expect(body).not.toContain('role="progressbar"');
	});
	it("does not show a fake completed bar when there are no areas", () => {
		const { body } = render(LiveStatus, { props: {
			learning: "manual", prediction: "waiting", progress: { ready: 0, total: 0 },
		} });
		expect(body).toContain("Needs a model");
		expect(body).not.toContain('role="progressbar"');
		expect(body).not.toContain("NaN");
	});
	it("keeps errors visible without a misleading spinner", () => {
		const { body } = render(LiveStatus, { props: {
			learning: "error", prediction: "retrying", progress: { ready: 2, total: 4 },
		} });
		expect(body).toContain("Learning blocked");
		expect(body).toContain("Predictions retrying");
		expect(body).not.toContain("animate-spin");
	});
});
