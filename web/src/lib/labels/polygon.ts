/**
 * Drawing polygons on a plane: closing one by clicking its first point,
 * drawing freehand by dragging (a lasso), and filling one with the active
 * class or cutting it out of that class.
 *
 * Points are level-0 voxels along the plane's u and v axes, as in
 * `raster.ts`. How far apart they look depends on the zoom and on the
 * voxels' shape, so distances people aim by are CSS pixels, through a view's
 * `Scale`.
 */

import type { Plane } from "../viewer/tiles";
import { PlaneMask } from "./raster";

export type Point = [number, number];

/** CSS pixels per level-0 voxel along the plane's u and v axes. */
export type Scale = [number, number];

/** A polygon being drawn: its plane, slice, and points. */
export interface Polygon {
	plane: Plane["name"];
	slice: number;
	points: Point[];
}

/** Closing a polygon fills it with the active class, or cuts it out of that class. */
export type PolygonMode = "add" | "subtract";

/** A click this close to a polygon's first point (CSS pixels) closes it. */
export const CLOSE_PIXELS = 8;
/**
 * A press that moves this far (CSS pixels) is a drag, which draws freehand
 * and closes on release; well past a click's wobble, since an accidental
 * drag would close the polygon being clicked out.
 */
export const DRAG_PIXELS = 8;
/** A freehand drag drops a point each time the pointer has gone this far (CSS pixels). */
export const LASSO_SPACING = 4;
// The server takes 16 KiB of tool record, so longer outlines aren't recorded.
const MAX_RECORDED_POINTS = 500;

/** How far apart two plane points are on screen, in CSS pixels. */
export function apart(a: Point, b: Point, scale: Scale): number {
	return Math.hypot((a[0] - b[0]) * scale[0], (a[1] - b[1]) * scale[1]);
}

/** Whether a click at `at` closes a polygon of `points`: it has three, and `at` is on the first. */
export function closesAt(points: readonly Point[], at: Point, scale: Scale, within = CLOSE_PIXELS): boolean {
	return points.length >= 3 && apart(points[0]!, at, scale) <= within;
}

/**
 * The points a freehand drag drops as the pointer passes `samples`, after
 * the polygon's last point (if it has one): each at least `spacing` CSS
 * pixels from the one before, so the outline follows the pointer smoothly
 * without a point per pixel.
 */
export function lassoPoints(last: Point | undefined, samples: readonly Point[], scale: Scale, spacing = LASSO_SPACING): Point[] {
	const dropped: Point[] = [];
	let previous = last;
	for (const sample of samples) {
		if (previous && apart(previous, sample, scale) < spacing) continue;
		dropped.push(sample);
		previous = sample;
	}
	return dropped;
}

/** What closing a polygon does with these keys held: Alt cuts out and Shift fills, whatever the mode. */
export function closingMode(mode: PolygonMode, keys: { altKey: boolean; shiftKey: boolean }): PolygonMode {
	if (keys.altKey) return "subtract";
	return keys.shiftKey ? "add" : mode;
}

/** A closed polygon as one label edit: the voxels it covers, what it writes there, and how it was drawn. */
export interface PolygonEdit {
	mask: PlaneMask;
	value: number;
	onlyIf: string;
	tool: Record<string, unknown>;
}

/**
 * The edit closing `polygon` makes, or null if it covers no voxel centers.
 * Filling writes the class `value` into the voxels `paintIf` (an `only_if`,
 * see `modes.ts`) lets change. Cutting out clears `value` inside and
 * leaves other classes' voxels alone, so a hole cut in one class never takes
 * a neighbour's labels with it (the eraser clears everything); that is its
 * own rule, whatever `paintIf` says. `limits` is the image's size along u
 * and v.
 */
export function polygonEdit(
	polygon: Polygon,
	mode: PolygonMode,
	value: number,
	paintIf: string,
	limits: [number, number],
): PolygonEdit | null {
	const mask = new PlaneMask(...limits);
	mask.polygon(polygon.points);
	if (mask.count === 0) return null;
	const cut = mode === "subtract";
	const points = polygon.points.map(([u, v]) => [Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
	return {
		mask,
		value: cut ? 0 : value,
		onlyIf: cut ? `class:${value}` : paintIf,
		tool: {
			name: cut ? "polygon-erase" : "polygon",
			plane: polygon.plane,
			slice: polygon.slice,
			points: points.length <= MAX_RECORDED_POINTS ? points : undefined,
		},
	};
}
