import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { TrainingNotifications, type TrainingModel } from "./training-notifications.svelte";

vi.mock("./api", async (original) => ({ ...await original<typeof import("./api")>(), api: vi.fn() }));
const request = vi.mocked(api);
const model: TrainingModel = { id: "m1", name: "Bone model", status: "training", live: false, created_by: "me" };
const body = { plugin: "rf", params: { n_estimators: 100 } };
let tracker: TrainingNotifications;
let stored: Map<string, string>;
let storage: { getItem: (key: string) => string | null; setItem: (key: string, value: string) => void };

beforeEach(() => {
	vi.useFakeTimers();
	request.mockReset();
	request.mockResolvedValue(model);
	stored = new Map();
	storage = { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => { stored.set(key, value); } };
	tracker = new TrainingNotifications();
	tracker.start("me", storage);
});
afterEach(() => { tracker.stop(); vi.useRealTimers(); });

describe("training submission", () => {
	it("locks immediately, refuses duplicate submits, then stays locked until completion", async () => {
		let resolve!: (value: TrainingModel) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		const first = tracker.submit("p", body);
		expect(tracker.isTraining("p")).toBe(true);
		expect(tracker.notices).toEqual([]); // Do not claim success before the POST is accepted.
		expect(await tracker.submit("p", body)).toBeNull();
		expect(request).toHaveBeenCalledTimes(1);
		resolve(model);
		await first;
		expect(tracker.isTraining("p")).toBe(true);
		expect(tracker.notices[0]?.message).toBe("Model is training, you will get a notification when training is complete!");
		expect(await tracker.submit("p", body)).toBeNull();
		request.mockResolvedValue({ ...model, status: "ready" });
		await vi.advanceTimersByTimeAsync(3000);
		expect(tracker.isTraining("p")).toBe(false);
		expect(tracker.notices.map((n) => n.kind)).toEqual(["ready"]);
	});

	it("unlocks rejected requests without claiming that training started", async () => {
		request.mockRejectedValueOnce(new ApiError(409, "trained_model_quota_exceeded"));
		await expect(tracker.submit("p", body)).rejects.toThrow();
		expect(tracker.isTraining("p")).toBe(false);
		expect(tracker.pending).toEqual([]);
		expect(tracker.notices).toEqual([]);
		await tracker.submit("p", body);
		expect(tracker.isTraining("p")).toBe(true);
	});

	it("tracks independent projects and allows their requests independently", async () => {
		await tracker.submit("one", body);
		request.mockResolvedValue({ ...model, id: "m2" });
		await tracker.submit("two", body);
		expect(tracker.pending.map((job) => job.project)).toEqual(["one", "two"]);
	});

	it("handles a model already ready when the POST returns", async () => {
		request.mockResolvedValue({ ...model, status: "ready" });
		await tracker.submit("p", body);
		expect(tracker.isTraining("p")).toBe(false);
		expect(tracker.notices.map((n) => n.kind)).toEqual(["ready"]);
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).toHaveBeenCalledTimes(1);
	});
});

describe("app-wide completion tracking", () => {
	it("keeps completion until dismissed, replacing the starting toast without its old timeout", async () => {
		await tracker.submit("p", body);
		request.mockResolvedValue({ ...model, status: "ready" });
		await vi.advanceTimersByTimeAsync(3000);
		expect(tracker.notices[0]?.message).toBe("Training complete — Bone model is ready.");
		await vi.advanceTimersByTimeAsync(60000);
		expect(tracker.notices).toHaveLength(1);
		expect(request).toHaveBeenCalledTimes(2);
		tracker.dismiss("p/m1");
		expect(tracker.notices).toEqual([]);
	});

	it("auto-dismisses only the starting toast without stopping tracking", async () => {
		await tracker.submit("p", body);
		await vi.advanceTimersByTimeAsync(8100);
		expect(tracker.notices).toEqual([]);
		expect(tracker.isTraining("p")).toBe(true);
	});

	it.each([null, "Stopped."])("reports failure or cancellation truthfully: %s", async (error) => {
		await tracker.submit("p", body);
		request.mockResolvedValue({ ...model, status: "failed", error });
		await vi.advanceTimersByTimeAsync(3000);
		expect(tracker.isTraining("p")).toBe(false);
		expect(tracker.notices[0]?.kind).toBe("failed");
		expect(tracker.notices[0]?.message).toContain(error ? "stopped" : "failed");
	});

	it.each([403, 404])("stops tracking inaccessible/deleted models without a success toast: %s", async (status) => {
		await tracker.submit("p", body);
		request.mockRejectedValue(new ApiError(status, "Gone"));
		await vi.advanceTimersByTimeAsync(3000);
		expect(tracker.isTraining("p")).toBe(false);
		expect(tracker.notices[0]?.kind).toBe("unavailable");
	});

	it("retries connection failures without duplicate notices or claiming failure", async () => {
		await tracker.submit("p", body);
		request.mockRejectedValueOnce(new TypeError("Offline"));
		await vi.advanceTimersByTimeAsync(3000);
		expect(tracker.pending).toHaveLength(1);
		expect(tracker.notices[0]?.kind).toBe("started");
		request.mockResolvedValue({ ...model, status: "ready" });
		await vi.advanceTimersByTimeAsync(10000);
		expect(tracker.notices.map((notice) => notice.kind)).toEqual(["ready"]);
	});

	it("never overlaps status polls when a request is slow", async () => {
		await tracker.submit("p", body);
		request.mockImplementation(() => new Promise(() => {}));
		await vi.advanceTimersByTimeAsync(60000);
		expect(request).toHaveBeenCalledTimes(2); // One POST, one unresolved GET.
	});

	it("can discover own manual trainings, without old successes, Live cycles, or other people's jobs", () => {
		tracker.watch("p", { ...model, live: true });
		tracker.watch("p", { ...model, created_by: "someone-else" });
		tracker.watch("p", { ...model, status: "ready" });
		expect(tracker.pending).toEqual([]);
		tracker.watch("p", model);
		tracker.watch("p", model);
		expect(tracker.pending).toHaveLength(1);
		expect(tracker.notices).toEqual([]);
	});

	it("doesn't re-register a completed job from a stale page refresh", async () => {
		tracker.watch("p", model);
		request.mockResolvedValue({ ...model, status: "ready" });
		await vi.advanceTimersByTimeAsync(3000);
		tracker.watch("p", model);
		expect(tracker.pending).toEqual([]);
		expect(tracker.notices).toHaveLength(1);
	});
});

describe("refresh and account safety", () => {
	it("restores pending training after reload and discovers completion immediately", async () => {
		await tracker.submit("p", body);
		tracker.stop();
		tracker = new TrainingNotifications();
		request.mockResolvedValue({ ...model, status: "ready" });
		tracker.start("me", storage);
		expect(tracker.isTraining("p")).toBe(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(tracker.notices[0]?.kind).toBe("ready");
	});

	it("preserves completion notices through reload, including dismissal", async () => {
		request.mockResolvedValue({ ...model, status: "ready" });
		await tracker.submit("p", body);
		tracker.stop(); tracker.start("me", storage);
		expect(tracker.notices[0]?.kind).toBe("ready");
		tracker.dismiss("p/m1");
		tracker.stop(); tracker.start("me", storage);
		expect(tracker.notices).toEqual([]);
	});

	it("isolates stored tracking by user", async () => {
		await tracker.submit("p", body);
		tracker.start("other-user", storage);
		expect(tracker.pending).toEqual([]);
		expect(tracker.notices).toEqual([]);
		tracker.start("me", storage);
		expect(tracker.isTraining("p")).toBe(true);
	});

	it.each(["submit", "poll"])("ignores a late %s response after switching accounts", async (phase) => {
		if (phase === "poll") await tracker.submit("p", body);
		let resolve!: (value: TrainingModel) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		const submission = phase === "submit" ? tracker.submit("p", body) : Promise.resolve(null);
		if (phase === "poll") await vi.advanceTimersByTimeAsync(3000);
		tracker.start("other-user", storage);
		resolve({ ...model, status: "ready" });
		await submission;
		await vi.advanceTimersByTimeAsync(0);
		expect(tracker.pending).toEqual([]);
		expect(tracker.submitting).toEqual([]);
		expect(tracker.notices).toEqual([]);
	});

	it("works when browser storage is unavailable or corrupt", async () => {
		tracker.stop();
		tracker.start("me", { getItem: () => "{", setItem: () => { throw new Error("Full"); } });
		await tracker.submit("p", body);
		expect(tracker.isTraining("p")).toBe(true);
	});

	it("does not restore malformed paths or duplicate jobs from storage", () => {
		tracker.stop();
		stored.set("ml4paleo:training:me", JSON.stringify({ pending: [
			{ project: "../admin", model: "m", name: "bad" }, null,
			{ project: "p", model: "m1", name: "ok" }, { project: "p", model: "m1", name: "ok" },
		] }));
		tracker.start("me", storage);
		expect(tracker.pending).toEqual([{ project: "p", model: "m1", name: "ok" }]);
	});
});
