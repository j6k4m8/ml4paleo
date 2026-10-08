/**
 * Drawing polygons on a plane: closing one by clicking its first point.
 *
 * Points are level-0 voxels along the plane's u and v axes, as in
 * `raster.ts`. How far apart they look depends on the zoom and on the
 * voxels' shape, so distances people aim by are CSS pixels, through a view's
 * `Scale`.
 */

export type Point = [number, number];

/** CSS pixels per level-0 voxel along the plane's u and v axes. */
export type Scale = [number, number];

/** A click this close to a polygon's first point (CSS pixels) closes it. */
export const CLOSE_PIXELS = 8;

/** How far apart two plane points are on screen, in CSS pixels. */
export function apart(a: Point, b: Point, scale: Scale): number {
	return Math.hypot((a[0] - b[0]) * scale[0], (a[1] - b[1]) * scale[1]);
}

/** Whether a click at `at` closes a polygon of `points`: it has three, and `at` is on the first. */
export function closesAt(points: readonly Point[], at: Point, scale: Scale, within = CLOSE_PIXELS): boolean {
	return points.length >= 3 && apart(points[0]!, at, scale) <= within;
}

