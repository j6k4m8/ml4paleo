import { describe, expect, it } from "vitest";
import { activityModelHref, groupActivity, latestPredictions, unfinished } from "./pipelines";
import type { Pipeline } from "./types";

function pipeline(id: string, kind: string, status: Pipeline["status"], model: string | null): Pipeline {
	return { id, kind, status, progress: 0, jobs: 1, created_at: "2026-10-06T12:00:00Z", error: null, model_id: model, created_by: null };
}

describe("latestPredictions", () => {
	it("keeps each model's newest prediction", () => {
		const latest = latestPredictions([
			pipeline("5", "ingest", "running", null),
			pipeline("4", "prediction", "running", "a"),
			pipeline("3", "training", "succeeded", "b"),
			pipeline("2", "prediction", "failed", "a"),
			pipeline("1", "prediction", "cancelled", "b"),
		]);
		expect(Object.keys(latest).sort()).toEqual(["a", "b"]);
		expect(latest.a?.id).toBe("4");
		expect(latest.b?.id).toBe("1");
	});
});

describe("unfinished", () => {
	it("is true while a pipeline waits or runs", () => {
		const statuses: Pipeline["status"][] = ["waiting", "running", "succeeded", "failed", "cancelled"];
		expect(statuses.filter((s) => unfinished(pipeline("1", "prediction", s, "a")))).toEqual(["waiting", "running"]);
	});
});

describe("groupActivity", () => {
	it("collapses consecutive successful runs, retaining the latest date and model", () => {
		const runs = Array.from({ length: 15 }, (_, i) => pipeline(String(15 - i), "training", "succeeded", `model-${15 - i}`));
		const before = structuredClone(runs);
		const groups = groupActivity(runs);
		expect(groups).toEqual([{ latest: runs[0], count: 15 }]);
		expect(activityModelHref("scan", groups[0]!)).toBe("/p/scan/models#model-model-15");
		expect(runs).toEqual(before);
	});

	it("preserves chronology and does not group across other kinds of work", () => {
		const groups = groupActivity([
			pipeline("6", "meshes", "succeeded", null),
			pipeline("5", "meshes", "succeeded", null),
			pipeline("4", "training", "succeeded", "a"),
			pipeline("3", "training", "succeeded", "b"),
			pipeline("2", "segmentation", "succeeded", null),
			pipeline("1", "training", "succeeded", "c"),
		]);
		expect(groups.map((group) => [group.latest.id, group.count])).toEqual([["6", 2], ["4", 2], ["2", 1], ["1", 1]]);
		expect(activityModelHref("scan", groups[0]!)).toBeNull();
	});

	it("keeps every running, waiting, failed, and cancelled run visible", () => {
		const statuses: Pipeline["status"][] = ["succeeded", "running", "running", "waiting", "failed", "failed", "cancelled", "succeeded"];
		const runs = statuses.map((status, i) => pipeline(String(i), "training", status, null));
		const groups = groupActivity(runs);
		expect(groups.map((group) => group.latest)).toEqual(runs);
		expect(groups.every((group) => group.count === 1)).toBe(true);
	});

	it("never buries an error and regroups when a run finishes", () => {
		const newest = pipeline("3", "training", "running", "a");
		const older = pipeline("2", "training", "succeeded", "b");
		expect(groupActivity([newest, older])).toHaveLength(2);
		newest.status = "succeeded";
		expect(groupActivity([newest, older])[0]?.count).toBe(2);
		older.error = "Something needs attention";
		expect(groupActivity([newest, older])).toHaveLength(2);
	});

	it("handles empty activity and does not borrow an older run's model", () => {
		expect(groupActivity([])).toEqual([]);
		const [group] = groupActivity([pipeline("2", "training", "succeeded", null), pipeline("1", "training", "succeeded", "old")]);
		expect(activityModelHref("scan", group!)).toBeNull();
	});
});
