/**
 * A project's regions of interest: the cubes and slices people label for
 * training. Marking one complete says its unlabeled voxels are background.
 */

import { api } from "#lib/api.ts";
import type { Vec3 } from "./viewer/tiles";

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
	error = $state("");

	constructor(private projectId: string) {}

	get #base(): string {
		return `/api/projects/${this.projectId}/rois`;
	}

	async load(): Promise<void> {
		this.items = await api<Roi[]>(this.#base);
	}

	async add(bbox: Box, kind: Roi["kind"]): Promise<Roi | null> {
		try {
			const roi = await api<Roi>(this.#base, { body: { bbox, kind } });
			this.items = [...this.items, roi];
			this.error = "";
			return roi;
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			return null;
		}
	}

	async update(id: string, change: Partial<Pick<Roi, "status" | "split">>): Promise<void> {
		try {
			const roi = await api<Roi>(`${this.#base}/${id}`, { method: "PATCH", body: change });
			this.items = this.items.map((r) => (r.id === id ? roi : r));
			this.error = "";
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
		}
	}

	async remove(id: string): Promise<void> {
		try {
			await api(`${this.#base}/${id}`, { method: "DELETE" });
			this.items = this.items.filter((r) => r.id !== id);
			this.error = "";
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
		}
	}
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
