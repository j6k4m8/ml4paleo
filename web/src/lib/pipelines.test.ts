import { describe, expect, it } from "vitest";
import { latestPredictions, unfinished } from "./pipelines";
import type { Pipeline } from "./types";

function pipeline(id: string, kind: string, status: Pipeline["status"], model: string | null): Pipeline {
	return { id, kind, status, progress: 0, jobs: 1, created_at: "2026-10-06T12:00:00Z", error: null, model_id: model };
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
