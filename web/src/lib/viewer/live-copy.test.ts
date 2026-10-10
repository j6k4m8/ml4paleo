import { describe, expect, it } from "vitest";
import { learningHelp, learningSchedule, reviewSlice } from "./live-copy";
import { PLANES } from "./tiles";

describe("plain-language suggestion copy", () => {
	it("distinguishes paused, periodic and manual learning without promising automatic learning for every method", () => {
		expect(learningSchedule("debounced")).toContain("after you pause painting");
		expect(learningSchedule("periodic")).toContain("at intervals");
		expect(learningSchedule("manual")).toContain("does not learn automatically");
		expect(learningSchedule(undefined)).toContain("Loading");
	});
	it("gives a next action for a full saved-model allowance", () => {
		expect(learningHelp("The project's owner keeps as many trained models as their limits allow.")).toContain("free space on Models");
		expect(learningHelp("trained_model_quota_exceeded")).toContain("raise the model limit");
	});
	it.each(["Add a label class first.", "Label something first.", "There are no labeled voxels to train on", "Training needs labels of at least two classes (background counts)"])("explains missing examples: %s", (detail) => {
		expect(learningHelp(detail)).toBe("Paint a few examples of what you want to find and some Background, then try again.");
	});
	it("keeps unknown technical diagnostics out of the main explanation", () => {
		expect(learningHelp("pipeline xyz checkpoint mismatch")).toBe("Try again. If it still fails, open the error details below.");
	});
	it("names the precise slice in all three orientations for the action and confirmation", () => {
		expect(reviewSlice(PLANES.xy, 20)).toBe("XY slice (Z 20)");
		expect(reviewSlice(PLANES.xz, 31)).toBe("XZ slice (Y 31)");
		expect(reviewSlice(PLANES.yz, 12)).toBe("YZ slice (X 12)");
	});
});
