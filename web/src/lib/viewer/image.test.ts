import { describe, expect, it } from "vitest";
import * as zarr from "zarrita";
import { failure, httpStatus, shardedFetch } from "./image";

/**
 * A 4 × 4 uint8 array in one shard of four 2 × 2 chunks (chunk k holds the
 * value k + 1), served from memory. Every answer takes a few milliseconds
 * and stops when its request's signal aborts, as a browser's would.
 */
function shardedArray() {
	const metadata = {
		zarr_format: 3,
		node_type: "array",
		shape: [4, 4],
		data_type: "uint8",
		chunk_grid: { name: "regular", configuration: { chunk_shape: [4, 4] } },
		chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
		fill_value: 0,
		codecs: [
			{
				name: "sharding_indexed",
				configuration: {
					chunk_shape: [2, 2],
					codecs: [{ name: "bytes", configuration: { endian: "little" } }],
					index_codecs: [{ name: "bytes", configuration: { endian: "little" } }],
					index_location: "end",
				},
			},
		],
	};
	const shard = new Uint8Array(16 + 64);
	const index = new DataView(shard.buffer, 16);
	for (let k = 0; k < 4; k++) {
		shard.fill(k + 1, 4 * k, 4 * k + 4);
		index.setBigUint64(16 * k, BigInt(4 * k), true);
		index.setBigUint64(16 * k + 8, 4n, true);
	}
	const reads: string[] = [];
	const fetcher = (request: Request) =>
		new Promise<Response>((resolve, reject) => {
			const abort = () => reject(new DOMException("Aborted", "AbortError"));
			if (request.signal?.aborted) return abort();
			request.signal?.addEventListener("abort", abort);
			setTimeout(() => {
				const path = new URL(request.url).pathname;
				const range = request.headers.get("range") ?? "";
				reads.push(range ? `${path} ${range}` : path);
				if (path.endsWith("/zarr.json")) return resolve(new Response(JSON.stringify(metadata)));
				const suffix = /^bytes=-(\d+)$/.exec(range);
				const span = /^bytes=(\d+)-(\d+)$/.exec(range);
				const body = suffix
					? shard.slice(shard.length - Number(suffix[1]))
					: span
						? shard.slice(Number(span[1]), Number(span[2]) + 1)
						: shard;
				resolve(new Response(body, { status: range ? 206 : 200 }));
			}, 5);
		});
	return { fetcher, reads };
}

describe("shardedFetch", () => {
	it("reads a shard's index for every chunk waiting on it, even if the read that asked first is cancelled", async () => {
		const { fetcher, reads } = shardedArray();
		const store = new zarr.FetchStore("http://test/image/", { useSuffixRequest: true, fetch: shardedFetch(fetcher) });
		const array = await zarr.open.v3(zarr.root(store), { kind: "array" });
		const first = new AbortController();
		const cancelled = zarr.get(array, [zarr.slice(0, 2), zarr.slice(0, 2)], { opts: { signal: first.signal } });
		const kept = zarr.get(array, [zarr.slice(2, 4), zarr.slice(2, 4)], { opts: { signal: new AbortController().signal } });
		first.abort();
		await expect(cancelled).rejects.toThrow();
		expect(Array.from((await kept).data as Uint8Array)).toEqual([4, 4, 4, 4]);
		// One index read served both chunks.
		expect(reads.filter((r) => r.includes("bytes=-"))).toHaveLength(1);
	});

	it("reads ranges past the browser's HTTP cache, which would take them one at a time", async () => {
		const seen: Request[] = [];
		const fetcher = shardedFetch(async (request) => {
			seen.push(request);
			return new Response("");
		});
		await fetcher(new Request("http://test/image/zarr.json"));
		await fetcher(new Request("http://test/image/c/0/0", { headers: { Range: "bytes=0-3" } }));
		expect(seen.map((r) => r.cache)).toEqual(["default", "no-store"]);
		expect(seen[1]?.headers.get("range")).toBe("bytes=0-3");
	});

	it("still cancels chunk reads", async () => {
		const { fetcher, reads } = shardedArray();
		const store = new zarr.FetchStore("http://test/image/", { useSuffixRequest: true, fetch: shardedFetch(fetcher) });
		const array = await zarr.open.v3(zarr.root(store), { kind: "array" });
		await zarr.get(array, [zarr.slice(0, 2), zarr.slice(0, 2)]);
		const controller = new AbortController();
		const read = zarr.get(array, [zarr.slice(2, 4), zarr.slice(0, 2)], { opts: { signal: controller.signal } });
		controller.abort();
		await expect(read).rejects.toThrow();
		await new Promise((resolve) => setTimeout(resolve, 20));
		expect(reads.filter((r) => r.includes("bytes=8-11"))).toHaveLength(0);
	});
});

describe("httpStatus", () => {
	/** What opening an array throws when the server answers with `status`. */
	const opening = (status: number) =>
		zarr
			.open(zarr.root(new zarr.FetchStore("http://test/image/", { fetch: async () => new Response("no", { status, statusText: "No" }) })), { kind: "array" })
			.then(
				() => undefined,
				(error: unknown) => error,
			);

	it("reads the status out of the error zarrita throws for an answer that isn't a chunk", async () => {
		for (const status of [401, 403, 500, 503]) {
			const error = await opening(status);
			expect(error).toBeInstanceOf(Error);
			expect(httpStatus(error)).toBe(status);
		}
	});

	it("is passed on with the error, for the page to decide whether to try again", async () => {
		const answered = failure(await opening(503));
		expect(answered.status).toBe(503);
		expect(answered.error).toContain("503");
		expect(failure(new TypeError("Failed to fetch"))).toEqual({ error: "TypeError: Failed to fetch", status: undefined });
	});

	it("says 404 for an array the server has no metadata for, which zarrita reports as not found", async () => {
		const error = await zarr
			.open(zarr.root(new zarr.FetchStore("http://test/labels/class_5/", { fetch: async () => new Response("no", { status: 404 }) })), { kind: "array" })
			.then(
				() => undefined,
				(e: unknown) => e,
			);
		expect(error).toBeInstanceOf(Error);
		expect(httpStatus(error)).toBeUndefined();
		expect(failure(error).status).toBe(404);
		expect(failure(error).error).toContain("NotFoundError");
	});

	it("says nothing for failures that weren't an answer", () => {
		expect(httpStatus(new TypeError("Failed to fetch"))).toBeUndefined();
		expect(httpStatus(new DOMException("Aborted", "AbortError"))).toBeUndefined();
		expect(httpStatus("a string")).toBeUndefined();
		expect(httpStatus(undefined)).toBeUndefined();
	});
});
