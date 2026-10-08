import { afterEach, describe, expect, it, vi } from "vitest";
import type { Chunk } from "./chunks";
import { LoadError, labelLevelLoader, WorkerPool } from "./loader";
import type { Level } from "./tiles";

/** A worker that does nothing but remember what it was asked and let the test answer for it. */
class FakeWorker {
	static all: FakeWorker[] = [];
	onmessage: ((event: { data: unknown }) => void) | null = null;
	posted: { id: number }[] = [];
	constructor() {
		FakeWorker.all.push(this);
	}
	postMessage(message: { id: number }) {
		this.posted.push(message);
	}
	terminate() {}
}

const region: [[number, number], [number, number], [number, number]] = [
	[0, 1],
	[0, 1],
	[0, 1],
];

describe("WorkerPool", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		FakeWorker.all = [];
	});

	const loading = () => {
		vi.stubGlobal("Worker", FakeWorker);
		const pool = new WorkerPool(1);
		const load = pool.load({ url: "http://test/", path: "0", region }, new AbortController().signal);
		const worker = FakeWorker.all[0]!;
		return { load, answer: (data: object) => worker.onmessage?.({ data: { id: worker.posted[0]!.id, ...data } }) };
	};

	it("fails a load with the HTTP status the worker saw", async () => {
		const { load, answer } = loading();
		answer({ error: "Error: Unexpected response status 401 Unauthorized", status: 401 });
		const error = await load.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(LoadError);
		expect((error as LoadError).status).toBe(401);
		expect((error as LoadError).message).toContain("401");
	});

	it("fails a load that got no answer from the server without a status", async () => {
		const { load, answer } = loading();
		answer({ error: "TypeError: Failed to fetch" });
		const error = await load.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(LoadError);
		expect((error as LoadError).status).toBeUndefined();
	});

	it("hands back what a worker read", async () => {
		const { load, answer } = loading();
		const data = new Uint8Array(1);
		answer({ data, shape: [1, 1, 1], version: 7 });
		expect(await load).toEqual({ data, shape: [1, 1, 1], version: 7 });
	});

	it("hands back what a coarser level's chunk was made from apart from a version", async () => {
		const { load, answer } = loading();
		const data = new Uint8Array(1);
		answer({ data, shape: [1, 1, 1], pyramid: 12 });
		const chunk = await load;
		expect(chunk.pyramid).toBe(12);
		expect(chunk.version).toBeUndefined();
	});
});

describe("labelLevelLoader", () => {
	// The labels of a 200 × 130 × 70 image, and its two coarser levels.
	const levels: Level[] = [
		{ index: 0, path: "class", shape: [200, 130, 70], scale: [1, 1, 1] },
		{ index: 1, path: "class_1", shape: [100, 65, 35], scale: [2, 2, 2] },
		{ index: 2, path: "class_2", shape: [50, 33, 18], scale: [4, 4, 4] },
	];
	const chunk: Chunk = { data: new Uint8Array(1), shape: [1, 1, 1] };

	const loading = (answer: (request: unknown) => Promise<Chunk> = async () => chunk) => {
		const load = vi.fn(async (request: unknown, _signal: AbortSignal) => answer(request));
		const missing = vi.fn();
		const loader = labelLevelLoader({ load } as unknown as WorkerPool, "http://test/labels/", () => levels, missing);
		return { load, missing, loader, signal: new AbortController().signal };
	};

	it("reads full resolution chunks, ids cz/cy/cx, from the class array, as the labels were read before levels", async () => {
		const { loader, load, signal } = loading();
		expect(await loader("1/2/0", signal)).toBe(chunk);
		expect(load).toHaveBeenCalledWith(
			{
				url: "http://test/labels/",
				path: "class",
				region: [
					[64, 128],
					[128, 130],
					[0, 64],
				],
				derived: false,
			},
			signal,
		);
	});

	it("reads a coarser level's chunks, ids level/cz/cy/cx, from its array, cut to the level's shape, as made ones", async () => {
		const { loader, load, signal } = loading();
		await loader("1/1/1/0", signal);
		expect(load).toHaveBeenCalledWith(
			{
				url: "http://test/labels/",
				path: "class_1",
				region: [
					[64, 100],
					[64, 65],
					[0, 35],
				],
				derived: true,
			},
			signal,
		);
	});

	it("says a level the server hasn't is missing, and ends the load without an error", async () => {
		const { loader, load, missing, signal } = loading(async () => {
			throw new LoadError("NotFoundError: Not found: v3 array or group", 404);
		});
		const error = await loader("1/0/0/0", signal).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(DOMException);
		expect((error as DOMException).name).toBe("AbortError");
		expect(missing).toHaveBeenCalledTimes(1);
		expect(missing).toHaveBeenCalledWith(1);
		expect(load).toHaveBeenCalledTimes(1);
	});

	it("passes on every other failure, as it was", async () => {
		for (const status of [401, 500, 503, undefined]) {
			const failure = new LoadError("Error", status);
			const { loader, missing, signal } = loading(async () => {
				throw failure;
			});
			await expect(loader("1/0/0/0", signal)).rejects.toBe(failure);
			expect(missing).not.toHaveBeenCalled();
		}
	});

	it("doesn't call full resolution missing when the server answers 404 for it", async () => {
		const failure = new LoadError("NotFoundError", 404);
		const { loader, missing, signal } = loading(async () => {
			throw failure;
		});
		await expect(loader("0/0/0", signal)).rejects.toBe(failure);
		expect(missing).not.toHaveBeenCalled();
	});

	it("ends a load of a level that was cut meanwhile without asking the server", async () => {
		const { loader, load, signal } = loading();
		const error = await loader("5/0/0/0", signal).catch((e: unknown) => e);
		expect((error as DOMException).name).toBe("AbortError");
		expect(load).not.toHaveBeenCalled();
	});
});
