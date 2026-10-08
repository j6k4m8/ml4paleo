/**
 * An OME-Zarr image's levels, read from its metadata through the data
 * gateway.
 */

import * as zarr from "zarrita";
import type { Level } from "./tiles";

interface Multiscales {
	datasets: { path: string; coordinateTransformations: { type: string; scale?: number[] }[] }[];
}

/** An absolute URL for a gateway path, as zarrita's FetchStore needs. */
export function absolute(url: string): string {
	return new URL(url, globalThis.location?.href ?? "http://localhost/").href;
}

/**
 * A fetch for zarrita's FetchStore reading sharded arrays, where each chunk
 * is a byte range of its shard's one URL.
 *
 * Ranges skip the browser's HTTP cache, which lets one request per URL
 * through at a time, so a shard's chunks would load one by one (decoded
 * chunks are cached in memory instead, by `ChunkStore`). And shard indexes
 * (suffix ranges) are read whatever happens to the read that asked first:
 * zarrita reads each index once for every chunk of its shard, passing on the
 * first read's abort signal, so cancelling that one chunk would fail all the
 * others waiting for the index.
 */
export function shardedFetch(
	fetcher: (request: Request) => Promise<Response> = (request) => fetch(request),
): (request: Request) => Promise<Response> {
	return (request) => {
		const range = request.headers.get("range");
		if (!range) return fetcher(request);
		const index = range.startsWith("bytes=-");
		return fetcher(new Request(request, { cache: "no-store", ...(index ? { signal: null } : {}) }));
	};
}

/**
 * The HTTP status in the error zarrita throws for a response that is neither
 * a chunk nor a missing one (404), if `error` is that.
 */
export function httpStatus(error: unknown): number | undefined {
	const match = /Unexpected response status (\d{3})/.exec(error instanceof Error ? error.message : String(error));
	return match ? Number(match[1]) : undefined;
}

/**
 * What a decode worker tells the page about a read that failed: the error,
 * and the HTTP status the server answered with, if it did.
 */
export function failure(error: unknown): { error: string; status?: number } {
	return { error: String(error), status: httpStatus(error) };
}

/** Level-0 voxels per voxel of each level, from the levels' scales. */
export function levelFactors(multiscales: Multiscales): [number, number, number][] {
	const scales = multiscales.datasets.map((dataset) => {
		const scale = dataset.coordinateTransformations.find((t) => t.type === "scale")?.scale;
		if (!scale || scale.length !== 4) throw new Error(`Level ${dataset.path} has no c,z,y,x scale`);
		return scale.slice(1) as [number, number, number];
	});
	const base = scales[0];
	if (!base) throw new Error("The image has no levels");
	return scales.map((scale) => scale.map((s, axis) => Math.round(s / (base[axis] ?? 1))) as [number, number, number]);
}

export async function loadLevels(url: string, signal?: AbortSignal): Promise<Level[]> {
	const store = new zarr.FetchStore(absolute(url), { useSuffixRequest: true });
	const group = await zarr.open.v3(zarr.root(store), { kind: "group", signal });
	const ome = group.attrs.ome as { multiscales?: Multiscales[] } | undefined;
	const multiscales = ome?.multiscales?.[0];
	if (!multiscales) throw new Error("This isn't an OME-Zarr image");
	const factors = levelFactors(multiscales);
	return Promise.all(
		multiscales.datasets.map(async (dataset, index) => {
			const array = await zarr.open.v3(zarr.root(store).resolve(dataset.path), {
				kind: "array",
				signal,
			});
			const [, z, y, x] = array.shape;
			return {
				index,
				path: dataset.path,
				shape: [z ?? 1, y ?? 1, x ?? 1],
				scale: factors[index] ?? [1, 1, 1],
			} satisfies Level;
		}),
	);
}
