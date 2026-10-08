import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as zarr from "zarrita";
import { PATIENCE_MS, patientFetch, retryDelay } from "./patient";

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

describe("patientFetch", () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	/** A fetch answering with `statuses` in turn (the last for any more), `Retry-After` on each 503. */
	const answering = (statuses: number[], retryAfter = "1") => {
		const asked: number[] = [];
		const fetcher = vi.fn(async () => {
			asked.push(Date.now());
			const status = statuses[Math.min(asked.length, statuses.length) - 1]!;
			return new Response(status === 200 ? "chunk" : "busy", { status, headers: status === 503 ? { "Retry-After": retryAfter } : {} });
		});
		return { fetcher, asked };
	};
	const request = (signal?: AbortSignal) => new Request("http://test/class_1/c/0/0/0", { signal });
	const random = () => 0.5;

	it("passes on any answer but a 503 at once", async () => {
		for (const status of [200, 404, 500]) {
			const { fetcher } = answering([status]);
			const response = await patientFetch(fetcher, { random })(request());
			expect(response.status).toBe(status);
			expect(fetcher).toHaveBeenCalledTimes(1);
		}
	});

	it("asks again after the Retry-After and a little more, until the chunk comes", async () => {
		const { fetcher, asked } = answering([503, 503, 200]);
		const start = Date.now();
		const answer = patientFetch(fetcher, { random })(request());
		await vi.advanceTimersByTimeAsync(1249);
		expect(fetcher).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(1);
		expect(fetcher).toHaveBeenCalledTimes(2);
		await vi.advanceTimersByTimeAsync(1250);
		const response = await answer;
		expect(response.status).toBe(200);
		expect(await response.text()).toBe("chunk");
		expect(asked.map((t) => t - start)).toEqual([0, 1250, 2500]);
	});

	it("spreads chunks that were told to wait together", async () => {
		const { fetcher } = answering([503, 200]);
		const early = patientFetch(fetcher, { random: () => 0 })(request());
		await vi.advanceTimersByTimeAsync(1000);
		expect(fetcher).toHaveBeenCalledTimes(2);
		await early;
		const late = answering([503, 200]);
		const answer = patientFetch(late.fetcher, { random: () => 1 })(request());
		await vi.advanceTimersByTimeAsync(1000);
		expect(late.fetcher).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(500);
		expect(late.fetcher).toHaveBeenCalledTimes(2);
		await answer;
	});

	it("passes the 503 on once another wait would end past its patience", async () => {
		const { fetcher, asked } = answering([503], "10");
		const start = Date.now();
		const answer = patientFetch(fetcher, { patience: 30_000, random: () => 0 })(request());
		const done = vi.fn();
		void answer.then(done);
		await vi.advanceTimersByTimeAsync(29_999);
		expect(done).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(1);
		expect(done).toHaveBeenCalledTimes(1);
		const response = await answer;
		expect(response.status).toBe(503);
		// Asked at 0, 10 s, 20 s and 30 s; the wait after that would end at 40.
		expect(asked.map((t) => t - start)).toEqual([0, 10_000, 20_000, 30_000]);
		// Nothing more is asked afterwards.
		await vi.advanceTimersByTimeAsync(120_000);
		expect(fetcher).toHaveBeenCalledTimes(4);
	});

	it("counts the time the answers took against its patience", async () => {
		const fetcher = vi.fn(async () => {
			// Each answer takes 20 s to come.
			await new Promise((resolve) => setTimeout(resolve, 20_000));
			return new Response("busy", { status: 503, headers: { "Retry-After": "1" } });
		});
		const answer = patientFetch(fetcher, { patience: 30_000, random: () => 0 })(request());
		await vi.advanceTimersByTimeAsync(50_000);
		expect((await answer).status).toBe(503);
		// 20 s for the first answer and a wait of 1 s leaves room for a second, which comes at 41 s.
		expect(fetcher).toHaveBeenCalledTimes(2);
	});

	it("is patient for two minutes by default", async () => {
		const { fetcher } = answering([503], "2");
		const answer = patientFetch(fetcher, { random: () => 0 })(request());
		await vi.advanceTimersByTimeAsync(PATIENCE_MS + 10_000);
		expect((await answer).status).toBe(503);
		// Asked every two seconds, from 0 to 120 s.
		expect(fetcher).toHaveBeenCalledTimes(61);
	});

	it("stops waiting at once when the request is cancelled", async () => {
		const { fetcher } = answering([503, 200]);
		const controller = new AbortController();
		const answer = patientFetch(fetcher, { random })(request(controller.signal));
		const failed = answer.catch((error: unknown) => error);
		await vi.advanceTimersByTimeAsync(500);
		controller.abort();
		const error = await failed;
		expect(error).toBeInstanceOf(DOMException);
		expect((error as DOMException).name).toBe("AbortError");
		expect(vi.getTimerCount()).toBe(0);
		await vi.advanceTimersByTimeAsync(10_000);
		expect(fetcher).toHaveBeenCalledTimes(1);
	});

	it("lets zarrita read a chunk the server took a few asks to make, where it alone takes the 503 for an error", async () => {
		const metadata = {
			zarr_format: 3,
			node_type: "array",
			shape: [2, 2, 2],
			data_type: "uint8",
			chunk_grid: { name: "regular", configuration: { chunk_shape: [2, 2, 2] } },
			chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
			fill_value: 0,
			codecs: [{ name: "bytes" }],
		};
		let busy = 2;
		const fetcher = async (request: Request) => {
			if (request.url.endsWith("/zarr.json")) return new Response(JSON.stringify(metadata));
			if (busy-- > 0) return new Response("busy", { status: 503, headers: { "Retry-After": "1" } });
			return new Response(new Uint8Array(8).fill(7));
		};
		const read = (fetch: (request: Request) => Promise<Response>) =>
			zarr
				.open.v3(zarr.root(new zarr.FetchStore("http://test/class_1/", { fetch })), { kind: "array" })
				.then((array) => zarr.get(array, [null, null, null]));
		await expect(read(fetcher)).rejects.toThrow("Unexpected response status 503");
		busy = 2;
		const patient = read(patientFetch(fetcher, { random }));
		await vi.advanceTimersByTimeAsync(3000);
		expect(Array.from((await patient).data as Uint8Array)).toEqual(new Array(8).fill(7));
	});
});
