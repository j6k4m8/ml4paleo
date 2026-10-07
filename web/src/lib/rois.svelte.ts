/**
 * A project's regions of interest: the cubes and slices people label for
 * training. Marking one complete says its unlabeled voxels are background.
 */

import { ApiError, api, message } from "#lib/api.ts";
import type { Vec3 } from "./viewer/tiles";
import { whileVisible } from "./refresh";

export type RoiStatus = "open" | "complete" | "skipped";
export type Box = [number, number, number, number, number, number];

export interface Roi {
	id: string;
	/** Global (z0, y0, x0, z1, y1, x1), half-open. */
	bbox: Box;
	kind: "cube" | "slice";
	status: RoiStatus;
	split: "train" | "val";
	origin: string;
	score: number | null;
	created_at: string;
}

export class RoiList {
	items = $state<Roi[]>([]);
	loaded = $state(false);
	error = $state("");

	constructor(private projectId: string) {}

	get #base(): string {
		return `/api/projects/${this.projectId}/rois`;
	}

	async load(): Promise<void> {
		try {
			this.items = await api<Roi[]>(this.#base);
			this.error = "";
		} catch (e) {
			this.error = message(e);
		} finally {
			this.loaded = true;
		}
	}

	/** Reload while the page is visible, and when it comes back into view. */
	keepFresh(): () => void {
		return whileVisible(() => void this.load());
	}

	/** Note a failure; an ROI someone else deleted leaves the list. */
	#failed(e: unknown, id?: string): void {
		if (id && e instanceof ApiError && e.status === 404) this.items = this.items.filter((r) => r.id !== id);
		this.error = message(e);
	}

	/** Make an ROI at a random place no ROI covers yet (see `exploreRoi`). */
	async explore(): Promise<Roi | null> {
		try {
			const roi = await exploreRoi(this.projectId);
			this.items = [...this.items, roi];
			this.error = "";
			return roi;
		} catch (e) {
			this.#failed(e);
			return null;
		}
	}

	async add(bbox: Box, kind: Roi["kind"]): Promise<Roi | null> {
		try {
			const roi = await api<Roi>(this.#base, { body: { bbox, kind } });
			this.items = [...this.items, roi];
			this.error = "";
			return roi;
		} catch (e) {
			this.#failed(e);
			return null;
		}
	}

	/** Change an ROI; returns whether the server took the change. */
	async update(id: string, change: Partial<Pick<Roi, "status" | "split">>): Promise<boolean> {
		try {
			const roi = await api<Roi>(`${this.#base}/${id}`, { method: "PATCH", body: change });
			this.items = this.items.map((r) => (r.id === id ? roi : r));
			this.error = "";
			return true;
		} catch (e) {
			this.#failed(e, id);
			return false;
		}
	}

	async remove(id: string): Promise<void> {
		try {
			await api(`${this.#base}/${id}`, { method: "DELETE" });
			this.items = this.items.filter((r) => r.id !== id);
			this.error = "";
		} catch (e) {
			this.#failed(e, id);
		}
	}
}

/**
 * Make an open ROI at a random place no ROI covers yet: a cube up to 128
 * voxels a side, small enough for a quick proposal.
 */
export function exploreRoi(projectId: string): Promise<Roi> {
	return api<Roi>(`/api/projects/${projectId}/rois/explore`, { method: "POST" });
}

/**
 * Make an ROI somewhere new and, if a model is ready, have the newest one
 * propose labels there, unless its prediction of the whole image already
 * shows there. The annotator shows the proposal once it's made.
 */
export async function explore(projectId: string): Promise<Roi> {
	const roi = await exploreRoi(projectId);
	const base = `/api/projects/${projectId}`;
	try {
		const models = await api<{ id: string; status: string }[]>(`${base}/models`);
		// Newest first.
		const newest = models.find((m) => m.status === "ready");
		const prediction = await api<{ model_id: string | null }>(`${base}/prediction`).catch(() => null);
		if (newest && prediction?.model_id !== newest.id) {
			await api(`${base}/models/${newest.id}/propose`, { body: { roi_id: roi.id } });
		}
	} catch {
		// The annotator offers "Propose here" instead.
	}
	return roi;
}

/** A short description of an ROI, such as "slice 40 × 30 × 1". */
export function describe(roi: Pick<Roi, "kind" | "bbox">): string {
	const [z0, y0, x0, z1, y1, x1] = roi.bbox;
	return `${roi.kind} ${x1 - x0} × ${y1 - y0} × ${z1 - z0}`;
}

/** Set a status or split select back to what the server has, after a refused change. */
export function revert(select: HTMLSelectElement, value: string): void {
	select.value = value;
}

/**
 * A box cut to an image's (z, y, x) shape, or null if nothing is left (an
 * ROI drawn on a bigger image that this one replaced may reach outside it).
 */
export function clipBox(box: Box, shape: readonly number[]): Box | null {
	const lo = [0, 1, 2].map((a) => Math.min(Math.max(box[a]!, 0), shape[a]!));
	const hi = [0, 1, 2].map((a) => Math.min(Math.max(box[a + 3]!, 0), shape[a]!));
	if (lo.some((l, a) => hi[a]! <= l)) return null;
	return [lo[0]!, lo[1]!, lo[2]!, hi[0]!, hi[1]!, hi[2]!];
}

/** The number of voxels in a box. */
export function voxels(box: Box): number {
	return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2]);
}

/** Whether two boxes share any voxel. */
export function overlaps(a: Box, b: readonly number[]): boolean {
	return [0, 1, 2].every((i) => a[i]! < b[i + 3]! && b[i]! < a[i + 3]!);
}

/** Whether box `inner` lies inside box `outer`. */
export function within(inner: Box, outer: readonly number[]): boolean {
	return [0, 1, 2].every((a) => outer[a]! <= inner[a]! && inner[a + 3]! <= outer[a + 3]!);
}

/** The thin axis of a slice ROI (or z for a cube). */
export function thinAxis(bbox: Box): 0 | 1 | 2 {
	const sizes = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]];
	const thinnest = sizes.indexOf(Math.min(...sizes));
	return sizes[thinnest] === 1 ? (thinnest as 0 | 1 | 2) : 0;
}

/**
 * The box for an ROI drawn as a rectangle on a plane: `depth` voxels along
 * the normal, centered on `slice` and moved to fit inside the image (1 for a
 * slice ROI).
 */
export function roiBox(
	plane: { normal: number; u: number; v: number },
	slice: number,
	corners: [[number, number], [number, number]],
	depth: number,
	shape: Vec3,
): Box | null {
	const lo: number[] = [0, 0, 0];
	const hi: number[] = [0, 0, 0];
	const [[ua, va], [ub, vb]] = corners;
	lo[plane.u] = Math.max(0, Math.floor(Math.min(ua, ub)));
	hi[plane.u] = Math.min(shape[plane.u]!, Math.ceil(Math.max(ua, ub)));
	lo[plane.v] = Math.max(0, Math.floor(Math.min(va, vb)));
	hi[plane.v] = Math.min(shape[plane.v]!, Math.ceil(Math.max(va, vb)));
	const n = plane.normal;
	const thickness = Math.max(1, Math.min(Math.round(depth), shape[n]!));
	let start = slice - Math.floor(thickness / 2);
	start = Math.max(0, Math.min(start, shape[n]! - thickness));
	lo[n] = start;
	hi[n] = start + thickness;
	if (lo.some((l, axis) => hi[axis]! <= l)) return null;
	return [lo[0]!, lo[1]!, lo[2]!, hi[0]!, hi[1]!, hi[2]!];
}
