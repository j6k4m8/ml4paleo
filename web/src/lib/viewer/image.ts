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
	return new URL(url, globalThis.location?.href).href;
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
	const store = new zarr.FetchStore(absolute(url));
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
				shape: [z ?? 1, y ?? 1, x ?? 1],
				scale: factors[index] ?? [1, 1, 1],
			} satisfies Level;
		}),
	);
}
