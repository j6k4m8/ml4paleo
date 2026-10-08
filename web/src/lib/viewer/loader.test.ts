import { afterEach, describe, expect, it, vi } from "vitest";
import { LoadError, WorkerPool } from "./loader";

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
});
