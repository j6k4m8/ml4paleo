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
 * A fetch for zarrita's FetchStore that reads shard indexes (the suffix
 * ranges it asks for) whatever happens to the read that asked first.
 * zarrita reads each shard's index once and shares it with every chunk of
 * the shard, passing on the first read's abort signal: cancelling that one
 * chunk would fail all the others waiting for the index.
 */
export function shardIndexesKept(
	fetcher: (request: Request) => Promise<Response> = (request) => fetch(request),
): (request: Request) => Promise<Response> {
	return (request) => {
		const suffix = request.headers.get("range")?.startsWith("bytes=-");
		return fetcher(suffix ? new Request(request, { signal: null }) : request);
	};
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
