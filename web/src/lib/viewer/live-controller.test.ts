import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "#lib/api.ts";
import { LivePreview } from "./live.svelte";
import type { LiveModel, ModelPlugin } from "./live";
import type { WorkerPool } from "./loader";

vi.mock("#lib/api.ts", () => ({ api: vi.fn(), message: (error: Error) => error.message }));
const request = vi.mocked(api);
const plugin: ModelPlugin = { name: "rf", version: "1", devices: ["cpu"], capabilities: {
	display_name: "Random forest", family: "classical", learning: "debounced", debounce_ms: 1000, min_train_interval_ms: 5000,
} };
const ready: LiveModel = { id: "old", name: "Old RF", plugin: "rf", status: "ready", live: false, created_by: "me", training_set: { image_artifact_id: "image", label_seq: 1 } };
let preview: LivePreview;
let frozen = false;
const decode = vi.fn();

beforeEach(() => {
	vi.useFakeTimers();
	vi.setSystemTime(100000);
	vi.stubGlobal("document", { hidden: false });
	frozen = false;
	decode.mockReset().mockResolvedValue({ data: Uint8Array.of(2), shape: [1, 1, 1] });
	preview = new LivePreview("p", "image", [64, 64, 128], { load: decode } as unknown as WorkerPool, "me", () => frozen);
	request.mockReset();
	request.mockResolvedValue([]);
});
afterEach(() => { preview.stop(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("live scheduling", () => {
	it("finishing learning retains the previous checkpoint's visible regions", async () => {
		request.mockResolvedValue([ready]);
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, learning: "manual" } });
		request.mockResolvedValue({ ready: true, artifact_id: "old-data", box: [0, 0, 0, 64, 64, 64], zarr_url: "/zarr/" });
		preview.enabled = true;
		preview.wanted = ["0/0/0"];
		await vi.advanceTimersByTimeAsync(500);
		const regions = preview.regions, store = preview.store;
		preview.changed(2);
		preview.plugin = plugin;
		request.mockImplementation(async (path) => {
			if (path.endsWith("/models")) return { ...ready, id: "new", live_current: true, training_set: { image_artifact_id: "image", label_seq: 2 } };
			return new Promise(() => {});
		});
		await vi.advanceTimersByTimeAsync(1500);
		expect(preview.model?.id).toBe("new");
		expect(preview.regions).toBe(regions);
		expect(preview.store).toBe(store);
		expect([...preview.stale]).toEqual(["0/0/0"]);
		expect(preview.updating).toBe(true);
		preview.predictionError = "Offline";
		expect(preview.updating).toBe(false);
		preview.predictionError = "";
		preview.paused = true;
		expect(preview.updating).toBe(false);
	});
	it("holds old decoded chunks until each replacement is downloaded, with its real provenance", async () => {
		request.mockResolvedValue([ready]);
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, learning: "manual" } });
		request.mockImplementation(async (path, options) => {
			const key = (options?.body as { key: number[] }).key;
			const x = key[2]! * 64;
			return { ready: true, artifact_id: `${path.includes('/new/') ? 'new' : 'old'}-${x}`, box: [0, 0, x, 64, 64, x + 64], zarr_url: '/zarr/' };
		});
		preview.enabled = true;
		preview.wanted = ["0/0/0", "0/0/1"];
		await vi.advanceTimersByTimeAsync(1000);
		const oldStore = preview.store!, oldRegions = preview.regions;
		const oldChunk = oldStore.peek("0/0/1");
		expect(oldRegions.size).toBe(2);
		expect(preview.progress).toEqual({ ready: 2, total: 2 });
		expect(preview.predictionPhase).toBe("ready");
		preview.changed(2);
		expect(preview.updating).toBe(false); // manual backend, no newer checkpoint yet
		preview.model = { ...ready, id: "new", name: "New RF", training_set: { image_artifact_id: "image", label_seq: 2 } };
		expect(preview.progress).toEqual({ ready: 0, total: 2 });
		expect(preview.predictionPhase).toBe("queued");
		expect(preview.updating).toBe(true);
		let replace!: (chunk: unknown) => void;
		decode.mockImplementationOnce(() => new Promise((resolve) => { replace = resolve; }));
		await vi.advanceTimersByTimeAsync(500);
		expect(preview.store).toBe(oldStore);
		expect([...preview.stale]).toEqual(["0/0/0", "0/0/1"]);
		expect(preview.regions.get("0/0/0")?.model_id).toBe("old");
		expect(preview.progress).toEqual({ ready: 0, total: 2 });
		expect(preview.predictionPhase).toBe("predicting");
		frozen = true;
		replace({ data: Uint8Array.of(3), shape: [1, 1, 1] });
		await vi.advanceTimersByTimeAsync(200);
		expect(preview.store).toBe(oldStore);
		frozen = false;
		await vi.advanceTimersByTimeAsync(100);
		expect(preview.regions.get("0/0/0")?.model_id).toBe("new");
		expect(preview.regions.get("0/0/1")).toBe(oldRegions.get("0/0/1"));
		expect(preview.store!.peek("0/0/0")?.data).toEqual(Uint8Array.of(3));
		expect(preview.store!.peek("0/0/1")).toBe(oldChunk);
		expect([...preview.stale]).toEqual(["0/0/1"]);
		expect(preview.versions.get("0/0/0")).toBe("new-0");
		expect(preview.progress).toEqual({ ready: 1, total: 2 });
		// The frozen routing snapshot used by an Accept gesture stays immutable.
		expect(oldStore.peek("0/0/0")?.data).toEqual(Uint8Array.of(2));
		decode.mockRejectedValueOnce(new Error("Download failed"));
		await vi.advanceTimersByTimeAsync(500);
		expect(preview.regions.get("0/0/1")).toBe(oldRegions.get("0/0/1"));
		expect(preview.predictionError).toBe("Download failed");
		expect(preview.predictionPhase).toBe("retrying");
		preview.wanted = ["0/0/0"];
		expect(preview.progress).toEqual({ ready: 1, total: 1 });
		expect(preview.predictionPhase).toBe("ready");
	});
	it("holds an in-flight prediction's publication throughout a drawn Accept gesture", async () => {
		request.mockResolvedValue([ready]);
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, learning: "manual" } });
		let resolve!: (value: unknown) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		preview.enabled = true;
		preview.wanted = ["0/0/0"];
		await vi.advanceTimersByTimeAsync(500);
		frozen = true;
		resolve({ ready: true, artifact_id: "next", box: [0, 0, 0, 64, 64, 64], zarr_url: "/zarr/" });
		await vi.advanceTimersByTimeAsync(3000);
		expect(preview.regions.size).toBe(0);
		expect(preview.store).toBeNull();
		frozen = false;
		await vi.advanceTimersByTimeAsync(100);
		expect(preview.regions.get("0/0/0")?.artifact_id).toBe("next");
		expect(preview.store).not.toBeNull();
	});

	it("does nothing until explicitly enabled, and pauses when hidden or accepting", async () => {
		await preview.select(plugin);
		request.mockClear();
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).not.toHaveBeenCalled();
		preview.enabled = true;
		preview.paused = true;
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).not.toHaveBeenCalled();
		preview.paused = false;
		frozen = true;
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).not.toHaveBeenCalled();
	});
	it("coalesces strokes and lets only one training request run", async () => {
		await preview.select(plugin);
		request.mockClear();
		request.mockImplementation(() => new Promise(() => {}));
		preview.enabled = true;
		preview.changed(1);
		await vi.advanceTimersByTimeAsync(500);
		preview.changed(2);
		await vi.advanceTimersByTimeAsync(500);
		expect(request).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).toHaveBeenCalledTimes(1);
		expect(request.mock.calls[0]?.[1]?.body).toEqual({ plugin: "rf", live: true });
	});
	it("keeps inference on the old checkpoint while learning, with no inference backlog", async () => {
		request.mockResolvedValue([ready]);
		await preview.select(plugin);
		request.mockClear();
		request.mockImplementation(() => new Promise(() => {}));
		preview.enabled = true;
		preview.wanted = ["0/0/0", "0/0/1"];
		await vi.advanceTimersByTimeAsync(10000);
		expect(request.mock.calls.map(([path]) => path)).toEqual(["/api/projects/p/models", "/api/projects/p/models/old/live-chunk"]);
		expect(preview.model?.id).toBe("old");
	});
	it("does not auto-train a manual backend", async () => {
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, family: "neural", learning: "manual" } });
		request.mockClear();
		preview.enabled = true;
		preview.changed(10);
		await vi.advanceTimersByTimeAsync(60000);
		expect(request).not.toHaveBeenCalled();
		expect(preview.learning).toBe("To learn from new labels, use the Models page.");
		expect(preview.learningPhase).toBe("manual");
		expect(preview.predictionPhase).toBe("waiting");
	});
	it("periodic learning obeys its interval and queues only the newest labels", async () => {
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, family: "neural", learning: "periodic", min_train_interval_ms: 30000 } });
		request.mockClear();
		request.mockResolvedValue({ ...ready, live_current: true });
		preview.enabled = true;
		preview.changed(2);
		await vi.advanceTimersByTimeAsync(1500);
		expect(request).toHaveBeenCalledTimes(1);
		preview.changed(3);
		preview.changed(4);
		await vi.advanceTimersByTimeAsync(29000);
		expect(request).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(1000);
		expect(request).toHaveBeenCalledTimes(2);
	});
	it("does not retrain forever after undo returns to an older pinned snapshot", async () => {
		await preview.select(plugin);
		request.mockClear();
		request.mockResolvedValue({ ...ready, live_current: true });
		preview.changed(15);
		preview.enabled = true;
		await vi.advanceTimersByTimeAsync(60000);
		expect(request).toHaveBeenCalledTimes(1);
	});
	it("refuses to publish a late checkpoint from a previous backend", async () => {
		await preview.select(plugin);
		let resolve!: (value: unknown) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		preview.enabled = true;
		await vi.advanceTimersByTimeAsync(500);
		request.mockResolvedValue([]);
		await preview.select({ ...plugin, name: "other" });
		resolve({ ...ready, live_current: true });
		await Promise.resolve();
		expect(preview.model).toBeNull();
	});
	it("drops finished chunks that navigation made obsolete", async () => {
		request.mockResolvedValue([ready]);
		await preview.select({ ...plugin, capabilities: { ...plugin.capabilities, learning: "manual" } });
		let resolve!: (value: unknown) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		preview.enabled = true;
		preview.wanted = ["0/0/0"];
		await vi.advanceTimersByTimeAsync(500);
		preview.wanted = ["0/0/1"];
		resolve({ ready: true, artifact_id: "outdated", box: [0, 0, 0, 64, 64, 64], zarr_url: "/zarr/" });
		await Promise.resolve();
		expect(preview.regions.size).toBe(0);
		expect(preview.store).toBeNull();
	});
	it("learning quota failures do not throttle inference on a ready checkpoint", async () => {
		request.mockResolvedValue([ready]);
		await preview.select(plugin);
		request.mockClear();
		request.mockImplementation(async (path) => {
			if (path.endsWith("/models")) throw new Error("Model quota reached");
			return { ready: true, artifact_id: "preview", box: [0, 0, 0, 64, 64, 64], zarr_url: "/zarr/" };
		});
		preview.enabled = true;
		preview.wanted = ["0/0/0", "0/0/1"];
		await vi.advanceTimersByTimeAsync(1500);
		expect(request.mock.calls.filter(([path]) => path.endsWith("live-chunk"))).toHaveLength(2);
		expect(preview.learningError).toBe("Model quota reached");
		expect(preview.learningPhase).toBe("error");
		expect(preview.regions.size).toBe(2);
		preview.changed(2);
		expect(preview.stale.size).toBe(2);
		expect(preview.updating).toBe(false);
	});

	it("uses plain statuses while learning and while suggestions are arriving", async () => {
		request.mockResolvedValue([ready]);
		await preview.select(plugin);
		expect(preview.learning).toBe("Ready to use previously learned examples.");
		request.mockImplementation(async (path) => {
			if (path.endsWith("/models")) return { ...ready, id: "new", status: "training", live_current: true };
			return new Promise(() => {});
		});
		preview.enabled = true;
		preview.wanted = ["0/0/0", "0/0/1"];
		await vi.advanceTimersByTimeAsync(500);
		expect(preview.learning).toBe("Learning from your saved labels…");
		expect(preview.predicting).toBe("Adding suggestions nearby… 0 of 2 areas ready.");
		expect(preview.learningPhase).toBe("learning");
		expect(preview.predictionPhase).toBe("predicting");
		expect(preview.progress).toEqual({ ready: 0, total: 2 });
		expect(preview.queuedEdits).toBe(false);
		preview.changed(2);
		expect(preview.queuedEdits).toBe(true);
	});

	it("distinguishes loading, queued learning, and finished learning without inventing progress", async () => {
		let resolve!: (value: unknown) => void;
		request.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
		const selecting = preview.select(plugin);
		expect(preview.learningPhase).toBe("loading");
		expect(preview.predictionPhase).toBe("waiting");
		expect(preview.progress).toEqual({ ready: 0, total: 0 });
		resolve([]);
		await selecting;
		expect(preview.learningPhase).toBe("queued");
		preview.changed(1);
		preview.enabled = true;
		request.mockResolvedValue({ ...ready, live_current: true });
		await vi.advanceTimersByTimeAsync(500);
		expect(preview.learningPhase).toBe("queued");
		await vi.advanceTimersByTimeAsync(500);
		expect(preview.learningPhase).toBe("ready");
		expect(preview.predictionPhase).toBe("waiting"); // No visible areas requested.
		expect(preview.queuedEdits).toBe(false);
	});

	it("retries failed learning without dropping usable suggestions and still respects hiding", async () => {
		request.mockResolvedValue([ready]);
		await preview.select(plugin);
		request.mockImplementation(async (path) => {
			if (path.endsWith("/models")) throw new Error("Model quota reached");
			return { ready: true, artifact_id: "preview", box: [0, 0, 0, 64, 64, 64], zarr_url: "/zarr/" };
		});
		preview.enabled = true;
		preview.wanted = ["0/0/0"];
		await vi.advanceTimersByTimeAsync(1500);
		const regions = preview.regions;
		const model = preview.model;
		preview.paused = true;
		request.mockClear();
		preview.retryLearning();
		expect(preview.learningError).toBe("");
		expect(preview.regions).toBe(regions);
		expect(preview.model).toBe(model);
		await vi.advanceTimersByTimeAsync(10000);
		expect(request).not.toHaveBeenCalled();
		preview.paused = false;
		await vi.advanceTimersByTimeAsync(500);
		expect(request).toHaveBeenCalledTimes(1);
		expect(request.mock.calls[0]?.[1]?.body).toEqual({ plugin: "rf", live: true });
	});
});
