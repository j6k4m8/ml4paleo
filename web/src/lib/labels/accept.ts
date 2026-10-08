/**
 * Accepting a model's prediction as labels: read the prediction inside a box
 * (an ROI, or the part of a slice a view shows) and turn it into edits, one
 * per predicted value (an op may touch a chunk only once), that fill only
 * voxels nobody has labeled yet.
 */

import type { Chunk } from "../viewer/chunks";
import type { Layout } from "../viewer/state.svelte";
import { type Plane, PLANES, TILE_HYSTERESIS, type Vec3 } from "../viewer/tiles";
import { CHUNK, type DeltaIn, splitIntoDeltas } from "./deltas";

/** Boxes bigger than this would make the page hold too much at once. */
export const MAX_ACCEPT_VOXELS = 256 ** 3;

/**
 * The most chunks (64³ voxels) a view draws the prediction from: with more
 * in view it's left out (see `PlaneView`), until the view needs this many
 * times fewer (`TILE_HYSTERESIS`).
 */
const MOST_DRAWN_CHUNKS = 128;

/**
 * The most chunks one accept in a view reads: as many as a view surely draws
 * the prediction from, so what's accepted is what's shown, and already
 * loaded. A view's slice has a chunk for each 64 × 64 of its voxels, so this
 * also holds it to far fewer voxels than the server takes at once (256³),
 * which is why only chunks are counted.
 */
export const MAX_VIEW_CHUNKS = Math.floor(MOST_DRAWN_CHUNKS / TILE_HYSTERESIS);

export type Box = [number, number, number, number, number, number];

/**
 * A box of a (z, y, x) uint8 array stored in 64³ chunks, assembled from the
 * chunks `load` gives by id (`cz/cy/cx`).
 */
export async function readBox(load: (id: string) => Promise<Chunk>, box: Box): Promise<Uint8Array> {
	const size = [box[3] - box[0], box[4] - box[1], box[5] - box[2]];
	const out = new Uint8Array(size[0]! * size[1]! * size[2]!);
	const first = [0, 1, 2].map((a) => Math.floor(box[a]! / CHUNK));
	const last = [0, 1, 2].map((a) => Math.floor((box[a + 3]! - 1) / CHUNK));
	const loads: Promise<void>[] = [];
	for (let cz = first[0]!; cz <= last[0]!; cz++) {
		for (let cy = first[1]!; cy <= last[1]!; cy++) {
			for (let cx = first[2]!; cx <= last[2]!; cx++) {
				const key = [cz, cy, cx];
				loads.push(
					load(key.join("/")).then((chunk) => {
						const [dz = 0, dy = 0, dx = 0] = chunk.shape;
						const data = chunk.data as unknown as ArrayLike<number>;
						const origin = key.map((k) => k * CHUNK);
						const lo = [0, 1, 2].map((a) => Math.max(box[a]!, origin[a]!));
						const hi = [0, 1, 2].map((a) => Math.min(box[a + 3]!, origin[a]! + [dz, dy, dx][a]!));
						for (let z = lo[0]!; z < hi[0]!; z++) {
							for (let y = lo[1]!; y < hi[1]!; y++) {
								const from = ((z - origin[0]!) * dy + (y - origin[1]!)) * dx - origin[2]!;
								const to = ((z - box[0]) * size[1]! + (y - box[1])) * size[2]! - box[2];
								for (let x = lo[2]!; x < hi[2]!; x++) out[to + x] = data[from + x]!;
							}
						}
					}),
				);
			}
		}
	}
	await Promise.all(loads);
	return out;
}

/**
 * The view that accepting "in this view" means: the only one, or in a
 * four-view layout the one the pointer is over, else the one last used.
 */
export function planeToAccept(layout: Layout, pointed: Plane | null, used: Plane): Plane {
	return layout === "four" ? (pointed ?? used) : PLANES[layout];
}

/** How many chunks (64³ voxels) a box touches. */
export function chunksIn(box: Box): number {
	let count = 1;
	for (let a = 0; a < 3; a++) count *= Math.floor((box[a + 3]! - 1) / CHUNK) - Math.floor(box[a]! / CHUNK) + 1;
	return count;
}

/** Whether the part of a slice a view shows takes more chunks than can be accepted at once. */
export function tooBigForView(box: Box): boolean {
	return chunksIn(box) > MAX_VIEW_CHUNKS;
}

/** What accepting in the view depends on. */
export interface ViewAccept {
	/** The project's image was replaced since this page opened. */
	imageReplaced: boolean;
	/** There's a prediction or a proposal (shown or not). */
	predicted: boolean;
	/** The prediction layer is on, and how opaque it is. */
	shown: boolean;
	opacity: number;
	/** The part of the slice the view shows, if it shows any. */
	box: Box | null;
	/** The box reaches over the edge of a proposal, so part of it shows the proposal and part doesn't. */
	mixed: boolean;
	/** Some layer shows over all of the box. */
	covered: boolean;
}

/** Why accepting in the view can't go ahead, in a sentence, or "" if it can. */
export function whyNotInView(now: ViewAccept): string {
	if (now.imageReplaced) return "This project's image was replaced; reload the page first.";
	if (!now.predicted) return "There's no prediction to accept yet.";
	if (!now.shown) return "The prediction is hidden; show it (M) to accept what's in view.";
	if (!(now.opacity > 0)) return "The prediction's opacity is 0; raise it to accept what's in view.";
	if (!now.box) return "Nothing of the image is in view.";
	if (tooBigForView(now.box)) return "Zoom in a bit: the visible area is too big to accept at once.";
	if (now.mixed) return "Part of this view shows your proposal and part doesn't; zoom in on one of them to accept it.";
	if (!now.covered) return "Nothing is predicted in this view yet.";
	return "";
}

/**
 * What a prediction holds over a box, with the voxels already labeled there
 * (`labeled`, the labels over the same box, 0 where there are none) taken
 * out: what an accept would fill. The server leaves labeled voxels alone
 * whatever it is sent; this keeps from sending them.
 */
export function unlabeledOnly(predicted: Uint8Array, labeled: Uint8Array): Uint8Array {
	const out = new Uint8Array(predicted.length);
	for (let i = 0; i < out.length; i++) if (!labeled[i]) out[i] = predicted[i]!;
	return out;
}

/** Edits that write each predicted value into the box's unlabeled voxels. */
export function acceptParts(values: Uint8Array, box: Box): DeltaIn[][] {
	const shape: Vec3 = [box[3] - box[0], box[4] - box[1], box[5] - box[2]];
	const origin: Vec3 = [box[0], box[1], box[2]];
	const present = new Set<number>();
	for (const value of values) if (value) present.add(value);
	return [...present]
		.sort((a, b) => a - b)
		.map((value) => {
			const mask = new Uint8Array(values.length);
			for (let i = 0; i < values.length; i++) if (values[i] === value) mask[i] = 1;
			return splitIntoDeltas(mask, shape, origin, { value, onlyIf: "unlabeled" });
		});
}
