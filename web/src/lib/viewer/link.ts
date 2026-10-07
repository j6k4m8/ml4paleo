/**
 * Links that open the annotator on a place: an ROI (`roi=<id>`) or a box
 * (`box=z0,y0,x0,z1,y1,x1`, whole voxels, half-open like ROIs' boxes). With
 * both, the annotator goes to the ROI if it's still there, else the box.
 */

import type { Box } from "../rois.svelte";

export function annotatorHref(projectId: string, place: { roi?: string | null; box?: Box | null } = {}): string {
	const query = [place.roi && `roi=${encodeURIComponent(place.roi)}`, place.box && `box=${place.box.join(",")}`]
		.filter(Boolean)
		.join("&");
	return `/p/${projectId}/annotate${query ? `?${query}` : ""}`;
}

/** The box a link gives, or null if it doesn't give one. */
export function parseBox(text: string | null): Box | null {
	const parts = text?.split(",") ?? [];
	if (parts.length !== 6 || !parts.every((part) => /^\d{1,9}$/.test(part))) return null;
	const box = parts.map(Number) as Box;
	return [0, 1, 2].every((axis) => box[axis]! < box[axis + 3]!) ? box : null;
}
