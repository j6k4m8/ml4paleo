import type { Box } from "../rois.svelte";
import type { Vec3 } from "./tiles";

export interface ModelPlugin {
	name: string;
	version: string;
	devices: string[];
	capabilities: {
		display_name: string;
		family: "classical" | "neural" | "other";
		learning: "manual" | "debounced" | "periodic";
		debounce_ms: number;
		min_train_interval_ms: number;
	};
}

export interface LiveModel {
	id: string;
	name: string;
	plugin: string;
	status: "training" | "ready" | "failed";
	error?: string | null;
	live: boolean;
	live_current?: boolean | null;
	created_by: string | null;
	training_set: { image_artifact_id: string; label_seq: number };
}

export function intersection(a: Box, b: Box): Box | null {
	const box = [0, 1, 2].map((axis) => Math.max(a[axis]!, b[axis]!))
		.concat([3, 4, 5].map((axis) => Math.min(a[axis]!, b[axis]!))) as Box;
	return [0, 1, 2].every((axis) => box[axis]! < box[axis + 3]!) ? box : null;
}

/** Bounded neighborhood: brush first, visible slices center-out, then nearby.
 * Never enumerate a whole volume or a zoomed-out screen's millions of tiles.
 */
export function liveKeys(shape: Vec3, center: Vec3, views: Box[], brush: Vec3 | null = null, limit = 128): string[] {
	const focus = center.map((n, i) => Math.max(0, Math.min(Math.ceil(shape[i]! / 64) - 1, Math.floor(n / 64)))) as Vec3;
	const keys: { id: string; rank: number; distance: number }[] = [];
	const brushKey = brush?.map((n) => Math.floor(n / 64)).join("/");
	for (let z = Math.max(0, focus[0] - 4); z <= Math.min(Math.ceil(shape[0] / 64) - 1, focus[0] + 4); z++) {
		for (let y = Math.max(0, focus[1] - 4); y <= Math.min(Math.ceil(shape[1] / 64) - 1, focus[1] + 4); y++) {
			for (let x = Math.max(0, focus[2] - 4); x <= Math.min(Math.ceil(shape[2] / 64) - 1, focus[2] + 4); x++) {
				const id = `${z}/${y}/${x}`;
				const box: Box = [z * 64, y * 64, x * 64, Math.min((z + 1) * 64, shape[0]), Math.min((y + 1) * 64, shape[1]), Math.min((x + 1) * 64, shape[2])];
				keys.push({ id, rank: id === brushKey ? -1 : views.some((view) => intersection(view, box)) ? 0 : 1,
					distance: (z - focus[0]) ** 2 + (y - focus[1]) ** 2 + (x - focus[2]) ** 2 });
			}
		}
	}
	keys.sort((a, b) => a.rank - b.rank || a.distance - b.distance || a.id.localeCompare(b.id));
	const result = keys.slice(0, limit).map((key) => key.id);
	// A stroke can be near the edge of a wide screen, outside the neighborhood.
	if (brushKey && !result.includes(brushKey) && brush!.every((n, i) => n >= 0 && n < shape[i]!)) result.unshift(brushKey);
	return result.slice(0, limit);
}
