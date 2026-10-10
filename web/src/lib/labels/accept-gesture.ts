/** Accept a drawn mask from immutable prediction sources, not a paint class. */
import { classOpacity, type ClassStyles } from "../viewer/class-display";
import { intersection } from "../viewer/live";
import type { Plane } from "../viewer/tiles";
import { acceptParts, chunksIn, MAX_ACCEPT_VOXELS, MAX_VIEW_CHUNKS, type Box } from "./accept";
import type { GroupedEdit } from "./opqueue.svelte";
import type { PlaneMask } from "./raster";

export interface GestureSource {
	artifact: string;
	box: Box;
	read: (box: Box) => Promise<Uint8Array>;
}

export function gestureBox(plane: Plane, slice: number, mask: PlaneMask): Box {
	const box: Box = [0, 0, 0, 0, 0, 0];
	box[plane.normal] = slice;
	box[plane.normal + 3] = slice + 1;
	box[plane.u] = mask.u0;
	box[plane.u + 3] = mask.u0 + mask.width;
	box[plane.v] = mask.v0;
	box[plane.v + 3] = mask.v0 + mask.height;
	return box;
}

/** Prepare the entire gesture before enqueuing anything; a failed chunk cannot cause a partial edit. */
export async function acceptGestureGroups(
	plane: Plane,
	slice: number,
	mask: PlaneMask,
	sources: readonly GestureSource[],
	readLabels: (box: Box) => Promise<Uint8Array>,
	styles: ClassStyles,
	classes: ReadonlySet<number>,
	tool: Record<string, unknown>,
): Promise<{ groups: GroupedEdit[]; count: number }> {
	const groups: GroupedEdit[] = [];
	let count = 0;
	if (!mask.count) return { groups, count };
	const bounds = gestureBox(plane, slice, mask);
	if (mask.width * mask.height > MAX_ACCEPT_VOXELS || chunksIn(bounds) > MAX_VIEW_CHUNKS) {
		throw new Error("That selection is too big to accept at once; draw a smaller one.");
	}
	for (const source of sources) {
		const box = intersection(bounds, source.box);
		if (!box) continue;
		const [predicted, labeled] = await Promise.all([source.read(box), readLabels(box)]);
		const size = (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2]);
		if (predicted.length !== size || labeled.length !== size) throw new Error("Incomplete prediction data; try again.");
		const values = new Uint8Array(size);
		let i = 0;
		for (let z = box[0]; z < box[3]; z++) {
			for (let y = box[1]; y < box[4]; y++) {
				for (let x = box[2]; x < box[5]; x++, i++) {
					const value = predicted[i]!;
					if (value <= 1 || value >= 255 || labeled[i] || !classes.has(value) || classOpacity(value, styles) === 0) continue;
					const point = [z, y, x];
					if (!mask.has(point[plane.u]!, point[plane.v]!)) continue;
					values[i] = value;
					count++;
				}
			}
		}
		const parts = acceptParts(values, box);
		if (parts.length) groups.push({ parts, options: { accept: { prediction: source.artifact, box }, tool } });
	}
	return { groups, count };
}
