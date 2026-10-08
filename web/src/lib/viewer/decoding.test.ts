import { describe, expect, it, vi } from "vitest";
import { chunkUrl, createDecoder, type DecodeRequest, type DecodeResponse } from "./decoding";

const BASE = "http://test/labels/";

/** The metadata of a uint8 zarr array of 64³ chunks. */
function metadata(shape: number[]) {
	return {
		zarr_format: 3,
		node_type: "array",
		shape,
		data_type: "uint8",
		chunk_grid: { name: "regular", configuration: { chunk_shape: [64, 64, 64] } },
		chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
		fill_value: 0,
		codecs: [{ name: "bytes" }],
	};
}

type Answer = { status?: number; headers?: Record<string, string>; value?: number };

/**
 * A server of the arrays `class_1` and `class_2` (64 × 64 × 128 voxels, so two chunks each) that answers
 * for each chunk URL with the latest `answers` says, a chunk of `value`s by default, and counts its requests.
 */
function serving(answers: Record<string, Answer> = {}) {
	const asked: string[] = [];
	const fetcher = vi.fn(async (request: Request) => {
		asked.push(request.url);
		if (request.url.endsWith("/zarr.json")) {
			const fail = answers[request.url];
			if (fail?.status) return new Response("no", { status: fail.status });
			return new Response(JSON.stringify(metadata([64, 64, 128])));
		}
		const { status = 200, headers = {}, value = 7 } = answers[request.url] ?? {};
		return new Response(status === 200 ? new Uint8Array(64 * 64 * 64).fill(value) : "busy", { status, headers });
	});
	return { fetcher, asked, answers };
}

const read = (path: string, extra: Partial<DecodeRequest> = {}): DecodeRequest => ({
	type: "load",
	id: 1,
	url: BASE,
	path,
	region: [
		[0, 64],
		[0, 64],
		[64, 128],
	],
	...extra,
});
const chunk = (path: string) => `${BASE}${path}/c/0/0/1`;
/** The values a reply carries. */
function valuesOf(reply: DecodeResponse): Uint8Array {
	if ("error" in reply) throw new Error(`The read failed: ${reply.error}`);
	return reply.data as Uint8Array;
}
const never = new AbortController().signal;

describe("chunkUrl", () => {
	it("names the chunk a region starts in", () => {
		expect(chunkUrl({ url: BASE, path: "class_2", region: [[64, 100], [128, 130], [0, 64]] })).toBe(`${BASE}class_2/c/1/2/0`);
		expect(chunkUrl(read("class"))).toBe(chunk("class"));
	});
});

describe("a decoder", () => {
	it("reads a region and says the version the server sent the chunk at, for an edit's base", async () => {
		const { fetcher } = serving({ [chunk("class_1")]: { headers: { "x-chunk-version": "5" }, value: 3 } });
		const { reply, transfer } = await createDecoder(fetcher)(read("class_1"), never);
		expect(reply).toMatchObject({ id: 1, shape: [64, 64, 64], version: 5, pyramid: undefined });
		const data = valuesOf(reply);
		expect(data).toHaveLength(64 * 64 * 64);
		expect(data[0]).toBe(3);
		expect(transfer).toEqual([data.buffer]);
	});

	it("says what a made chunk was made from instead of a version, whatever else the server sent with it", async () => {
		const { fetcher } = serving({ [chunk("class_1")]: { headers: { "x-chunk-version": "5", "x-pyramid-version": "12" } } });
		const { reply } = await createDecoder(fetcher)(read("class_1", { derived: true }), never);
		expect(reply).toMatchObject({ pyramid: 12, version: undefined });
	});

	it("takes a stored chunk's version, not what a made one was made from", async () => {
		const { fetcher } = serving({ [chunk("class")]: { headers: { "x-chunk-version": "5", "x-pyramid-version": "12" } } });
		const { reply } = await createDecoder(fetcher)(read("class"), never);
		expect(reply).toMatchObject({ version: 5, pyramid: undefined });
	});

	it("says neither for a chunk the server sent without them", async () => {
		const { fetcher } = serving();
		for (const derived of [false, true]) {
			const { reply } = await createDecoder(fetcher)(read("class_1", { derived }), never);
			expect(reply).toMatchObject({ version: undefined, pyramid: undefined });
		}
	});

	it("forgets what the server said of a chunk once it has been read", async () => {
		const { fetcher, answers } = serving({ [chunk("class_1")]: { headers: { "x-chunk-version": "5" } } });
		const decode = createDecoder(fetcher);
		expect((await decode(read("class_1"), never)).reply).toMatchObject({ version: 5 });
		answers[chunk("class_1")] = {};
		expect((await decode(read("class_1"), never)).reply).toMatchObject({ version: undefined });
	});

	it("keeps the arrays at different paths apart, and opens each once", async () => {
		const { fetcher, asked } = serving({ [chunk("class_1")]: { value: 1 }, [chunk("class_2")]: { value: 2 } });
		const decode = createDecoder(fetcher);
		const values = [];
		for (const path of ["class_1", "class_2", "class_1", "class_2"]) {
			const { reply } = await decode(read(path), never);
			values.push(valuesOf(reply)[0]);
		}
		expect(values).toEqual([1, 2, 1, 2]);
		expect(asked.filter((url) => url.endsWith("/zarr.json"))).toEqual([`${BASE}class_1/zarr.json`, `${BASE}class_2/zarr.json`]);
	});

	it("keeps the arrays of different stores apart too", async () => {
		const there = "http://elsewhere/labels/";
		const { fetcher } = serving({ [chunk("class_1")]: { value: 1 }, [`${there}class_1/c/0/0/1`]: { value: 2 } });
		const decode = createDecoder(fetcher);
		expect(valuesOf((await decode(read("class_1"), never)).reply)[0]).toBe(1);
		expect(valuesOf((await decode(read("class_1", { url: there }), never)).reply)[0]).toBe(2);
	});

	it("answers a 503 as it is, with its Retry-After, and doesn't ask again itself", async () => {
		const { fetcher, asked } = serving({ [chunk("class_1")]: { status: 503, headers: { "retry-after": "3" } } });
		const { reply, transfer } = await createDecoder(fetcher)(read("class_1", { derived: true }), never);
		expect(reply).toEqual({ id: 1, error: expect.stringContaining("503"), status: 503, retryAfter: "3" });
		expect(transfer).toEqual([]);
		expect(asked.filter((url) => url === chunk("class_1"))).toHaveLength(1);
	});

	it("says no Retry-After for a 503 that came without one", async () => {
		const { fetcher } = serving({ [chunk("class_1")]: { status: 503 } });
		const { reply } = await createDecoder(fetcher)(read("class_1", { derived: true }), never);
		expect(reply).toMatchObject({ status: 503 });
		expect(reply).not.toHaveProperty("retryAfter");
	});

	it("says a Retry-After only for the chunk it came with, and only once", async () => {
		const { fetcher, answers } = serving({ [chunk("class_1")]: { status: 503, headers: { "retry-after": "3" } } });
		const decode = createDecoder(fetcher);
		await decode(read("class_1"), never);
		// The same chunk fails another way later, and another chunk too: neither is told to wait.
		answers[chunk("class_1")] = { status: 500 };
		expect((await decode(read("class_1"), never)).reply).toEqual({ id: 1, error: expect.stringContaining("500"), status: 500 });
		const other = read("class_1", { region: [[0, 64], [0, 64], [0, 64]] });
		answers[`${BASE}class_1/c/0/0/0`] = { status: 500, headers: { "retry-after": "9" } };
		expect((await decode(other, never)).reply).toEqual({ id: 1, error: expect.stringContaining("500"), status: 500 });
	});

	it("says the status of any other answer the server gave, and 404 for an array it doesn't have", async () => {
		const { fetcher } = serving({ [chunk("class_1")]: { status: 401 }, [`${BASE}class_2/zarr.json`]: { status: 404 } });
		const decode = createDecoder(fetcher);
		expect((await decode(read("class_1"), never)).reply).toMatchObject({ status: 401 });
		expect((await decode(read("class_2"), never)).reply).toMatchObject({ status: 404 });
	});

	it("says no status for a failure to reach the server", async () => {
		const decode = createDecoder(async () => {
			throw new TypeError("Failed to fetch");
		});
		const { reply } = await decode(read("class_1"), never);
		expect(reply).toMatchObject({ id: 1, error: expect.stringContaining("Failed to fetch") });
		expect(reply).not.toHaveProperty("status", expect.anything());
	});

	it("opens an array again after failing to", async () => {
		const { fetcher, answers } = serving({ [`${BASE}class_1/zarr.json`]: { status: 500 } });
		const decode = createDecoder(fetcher);
		expect((await decode(read("class_1"), never)).reply).toMatchObject({ status: 500 });
		answers[`${BASE}class_1/zarr.json`] = {};
		expect((await decode(read("class_1"), never)).reply).toMatchObject({ shape: [64, 64, 64] });
	});

	it("stops reading when cancelled", async () => {
		const controller = new AbortController();
		const decode = createDecoder(async (request) => {
			if (request.url.endsWith("/zarr.json")) return new Response(JSON.stringify(metadata([64, 64, 128])));
			controller.abort();
			throw new DOMException("Aborted", "AbortError");
		});
		const { reply } = await decode(read("class_1"), controller.signal);
		expect(reply).toMatchObject({ error: expect.stringContaining("AbortError") });
	});
});
