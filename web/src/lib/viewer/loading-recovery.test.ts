import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChunkStore } from "./chunks";
import { LOAD_RETRIES, LoadError } from "./load-errors";
import { WorkerPool, WORKER_TIMEOUT_MS } from "./loader";
import { createDecoder, type DecodeRequest } from "./decoding";
import { READ_TIMEOUT_MS, timedFetch } from "./fetch";
import { failure, shardedFetch } from "./image";

const chunk = { data: new Uint8Array([2]), shape: [1, 1, 1] };
const region: DecodeRequest["region"] = [[0, 1], [0, 1], [0, 1]];
const input = { url: "http://test/", path: "class", region };
const request: DecodeRequest = { ...input, type: "load", id: 1 };
const metadata = { zarr_format: 3, node_type: "array", shape: [1, 1, 1], data_type: "uint8",
	chunk_grid: { name: "regular", configuration: { chunk_shape: [1, 1, 1] } },
	chunk_key_encoding: { name: "default", configuration: { separator: "/" } }, fill_value: 0, codecs: [{ name: "bytes" }] };

beforeEach(() => { vi.useFakeTimers(); vi.spyOn(Math, "random").mockReturnValue(0); });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("chunk retry scheduling", () => {
	it("retries a transient failure without holding up the next chunk or needing a redraw", async () => {
		const load = vi.fn().mockRejectedValueOnce(new LoadError("offline", undefined, undefined, true)).mockResolvedValue(chunk);
		const store = new ChunkStore(load, 1000, 1);
		store.want("xy", ["a", "b"]);
		const a = store.request("a"), b = store.request("b");
		await vi.advanceTimersByTimeAsync(0);
		expect(await b).toBe(chunk);
		expect(store.isLoading("a")).toBe(true);
		expect(load.mock.calls.map(([id]) => id)).toEqual(["a", "b"]);
		await vi.advanceTimersByTimeAsync(1000);
		expect(await a).toBe(chunk);
		expect(load.mock.calls.map(([id]) => id)).toEqual(["a", "b", "a"]);
		expect(vi.getTimerCount()).toBe(0);
	});
	it("bounds retries across redraws, and an explicit Retry recovers", async () => {
		const load = vi.fn().mockRejectedValue(new LoadError("server error", 500));
		const store = new ChunkStore(load, 1000);
		store.want("xy", ["a"]);
		const a = store.request("a").catch((e) => e);
		await vi.advanceTimersByTimeAsync(600_000);
		expect(await a).toBeInstanceOf(LoadError);
		expect(load).toHaveBeenCalledTimes(1 + LOAD_RETRIES);
		for (let i = 0; i < 30; i++) { store.want("xy", ["a"]); await store.request("a").catch(() => {}); }
		expect(load).toHaveBeenCalledTimes(1 + LOAD_RETRIES);
		expect(store.failed).toBe(true);
		load.mockResolvedValue(chunk);
		store.retryFailed();
		expect(await store.request("a")).toBe(chunk);
		expect(store.failed).toBe(false);
	});
	it("cancels delayed retries as soon as the view leaves", async () => {
		const load = vi.fn().mockRejectedValue(new LoadError("busy", 503));
		const store = new ChunkStore(load, 1000);
		store.want("xy", ["a"]);
		const a = store.request("a").catch((e) => e);
		await vi.advanceTimersByTimeAsync(0);
		store.want("xy", []);
		await vi.advanceTimersByTimeAsync(600_000);
		expect((await a).name).toBe("AbortError");
		expect(load).toHaveBeenCalledTimes(1);
		expect(vi.getTimerCount()).toBe(0);
	});
	it.each([400, 401, 403, 404, 422])("does not automatically retry permanent HTTP %s errors", async (status) => {
		const load = vi.fn().mockRejectedValue(new LoadError("no", status));
		const store = new ChunkStore(load, 1000);
		await expect(store.request("a")).rejects.toBeInstanceOf(LoadError);
		await vi.advanceTimersByTimeAsync(600_000);
		expect(load).toHaveBeenCalledTimes(1);
	});
	it("honors Retry-After on 429 and spaces simultaneous retries", async () => {
		const load = vi.fn().mockRejectedValueOnce(new LoadError("busy", 429, "10")).mockRejectedValueOnce(new LoadError("busy", 429, "10")).mockResolvedValue(chunk);
		const store = new ChunkStore(load, 1000);
		const a = store.request("a"), b = store.request("b");
		await vi.advanceTimersByTimeAsync(9999);
		expect(load).toHaveBeenCalledTimes(2);
		await vi.advanceTimersByTimeAsync(1);
		expect(await a).toBe(chunk);
		expect(load).toHaveBeenCalledTimes(3);
		await vi.advanceTimersByTimeAsync(250);
		expect(await b).toBe(chunk);
	});
	it("does not retry sooner when Retry-After exceeds the overall budget", async () => {
		const load = vi.fn().mockRejectedValue(new LoadError("busy", 429, "3600"));
		const store = new ChunkStore(load, 1000);
		await expect(store.request("a")).rejects.toBeInstanceOf(LoadError);
		await vi.advanceTimersByTimeAsync(600_000);
		expect(load).toHaveBeenCalledTimes(1);
	});
	it("doesn't multiply the label layer's existing refresh retries", async () => {
		const load = vi.fn().mockResolvedValueOnce(chunk).mockRejectedValue(new LoadError("busy", 500));
		const store = new ChunkStore(load, 1000);
		await store.request("a");
		await expect(store.refresh("a")).rejects.toBeInstanceOf(LoadError);
		await vi.advanceTimersByTimeAsync(600_000);
		expect(load).toHaveBeenCalledTimes(2);
		expect(store.peek("a")).toBe(chunk);
	});
});

class FakeWorker {
	static all: FakeWorker[] = [];
	onmessage: ((event: { data: unknown }) => void) | null = null;
	onerror: ((event: { preventDefault(): void }) => void) | null = null;
	onmessageerror: (() => void) | null = null;
	posted: { type: string; id: number }[] = [];
	terminate = vi.fn();
	postMessage = vi.fn((message: { type: string; id: number }) => { this.posted.push(message); });
	constructor() { FakeWorker.all.push(this); }
	answer(id = this.posted[0]!.id) { this.onmessage?.({ data: { id, ...chunk } }); }
}

describe("decode worker recovery", () => {
	beforeEach(() => { FakeWorker.all = []; vi.stubGlobal("Worker", FakeWorker); });
	it.each(["crash", "messageerror", "timeout"])("replaces a worker after %s, settles all its loads, and ignores late replies", async (kind) => {
		const pool = new WorkerPool(1), signal = new AbortController().signal;
		const a = pool.load(input, signal).catch((e) => e), b = pool.load(input, signal).catch((e) => e);
		const old = FakeWorker.all[0]!;
		if (kind === "crash") old.onerror?.({ preventDefault() {} });
		else if (kind === "messageerror") old.onmessageerror?.();
		else await vi.advanceTimersByTimeAsync(WORKER_TIMEOUT_MS);
		expect(await a).toMatchObject({ transient: true });
		expect(await b).toMatchObject({ transient: true });
		expect(old.terminate).toHaveBeenCalledOnce();
		expect(FakeWorker.all).toHaveLength(1); // Lazy respawn, not an error loop.
		let settled = false;
		const c = pool.load(input, signal).then((v) => { settled = true; return v; });
		const fresh = FakeWorker.all[1]!;
		old.answer(fresh.posted[0]!.id);
		await vi.advanceTimersByTimeAsync(0);
		expect(settled).toBe(false);
		fresh.answer();
		expect(await c).toEqual(chunk);
		pool.close();
		expect(vi.getTimerCount()).toBe(0);
	});
	it("cleans up cancellation, already-aborted requests, and close without respawning", async () => {
		const pool = new WorkerPool(1), abort = new AbortController();
		const a = pool.load(input, abort.signal).catch((e) => e);
		abort.abort();
		expect((await a).name).toBe("AbortError");
		await expect(pool.load(input, abort.signal)).rejects.toMatchObject({ name: "AbortError" });
		const worker = FakeWorker.all[0]!;
		expect(worker.posted.map((m) => m.type)).toEqual(["load", "cancel"]);
		const b = pool.load(input, new AbortController().signal).catch((e) => e);
		pool.close();
		expect((await b).name).toBe("AbortError");
		await expect(pool.load(input, new AbortController().signal)).rejects.toMatchObject({ name: "AbortError" });
		expect(vi.getTimerCount()).toBe(0);
		expect(FakeWorker.all).toHaveLength(1);
	});
	it("settles failed postMessage calls rather than leaking a waiting entry", async () => {
		const pool = new WorkerPool(1);
		FakeWorker.all[0]!.postMessage.mockImplementation(() => { throw new Error("worker gone"); });
		await expect(pool.load(input, new AbortController().signal)).rejects.toMatchObject({ transient: true });
		expect(vi.getTimerCount()).toBe(0);
		pool.close();
	});
});

describe("HTTP read deadlines", () => {
	beforeEach(() => {
		// Native AbortSignal.timeout doesn't use fake timers. Keep its behavior,
		// including its reason, while advancing time without a real 20s wait.
		vi.spyOn(AbortSignal, "timeout").mockImplementation((ms) => {
			const controller = new AbortController();
			setTimeout(() => controller.abort(new DOMException("Timed out", "TimeoutError")), ms);
			return controller.signal;
		});
	});
	it("expires a shared metadata read and lets a fresh request recover", async () => {
		let hanging = true;
		const fetcher = vi.fn((req: Request): Promise<Response> => {
			if (!req.url.endsWith("zarr.json")) return Promise.resolve(new Response(new Uint8Array([2])));
			if (!hanging) return Promise.resolve(new Response(JSON.stringify(metadata)));
			return new Promise((_resolve, reject) => req.signal.addEventListener("abort", () => reject(req.signal.reason), { once: true }));
		});
		const decode = createDecoder(fetcher);
		const first = new AbortController();
		const a = decode(request, first.signal), b = decode({ ...request, id: 2 }, new AbortController().signal);
		await vi.advanceTimersByTimeAsync(0);
		first.abort();
		expect(fetcher).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS);
		expect((await a).reply).toHaveProperty("error");
		expect((await b).reply).toMatchObject({ retryable: true });
		hanging = false;
		expect((await decode(request, new AbortController().signal)).reply).toMatchObject(chunk);
		expect(fetcher).toHaveBeenCalledTimes(3);
	});
	it("bounds suffix-index reads even though they outlive the first view's cancellation", async () => {
		let shared!: Request;
		const fetcher = shardedFetch(timedFetch(async (req) => { shared = req; return new Response(""); }));
		const cancelled = new AbortController();
		await fetcher(new Request("http://test/shard", { headers: { Range: "bytes=-16" }, signal: cancelled.signal }));
		cancelled.abort();
		expect(shared.signal.aborted).toBe(false);
		expect(shared.cache).toBe("no-store");
		await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS);
		expect(shared.signal.reason.name).toBe("TimeoutError");
	});
	it("times out a stalled response body too", async () => {
		const decode = createDecoder(async (req) => {
			if (req.url.endsWith("zarr.json")) return new Response(JSON.stringify(metadata));
			return new Response(new ReadableStream({ start(controller) {
				req.signal.addEventListener("abort", () => controller.error(new DOMException("Body aborted", "AbortError")), { once: true });
			} }));
		});
		const loading = decode(request, new AbortController().signal);
		await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS);
		expect((await loading).reply).toMatchObject({ retryable: true });
	});
	it("preserves Retry-After for every chunk sharing a metadata failure", async () => {
		const decode = createDecoder(async () => new Response("busy", { status: 429, headers: { "Retry-After": "15" } }));
		const results = await Promise.all([1, 2].map((id) => decode({ ...request, id }, new AbortController().signal)));
		for (const { reply } of results) expect(reply).toMatchObject({ status: 429, retryAfter: "15" });
	});
	it("does not mistake corrupt data or user cancellation for a transient network failure", () => {
		expect(failure(new TypeError("Invalid typed array length"))).not.toHaveProperty("retryable");
		expect(failure(new DOMException("Moved away", "AbortError"))).not.toHaveProperty("retryable");
		expect(failure(new TypeError("Load failed"))).toHaveProperty("retryable", true);
	});
});
