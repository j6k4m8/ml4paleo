/**
 * Which pyramid level and which chunks a plane view needs.
 *
 * Coordinates are in full-resolution (level 0) voxels, ordered (z, y, x) as
 * in storage. A view shows the plane through `position` normal to one axis,
 * centered on `position`; its horizontal axis is `u` and its vertical axis
 * `v`. `zoom` is screen pixels per level-0 voxel along the finest axis, and
 * `aspect` stretches coarser axes so voxels show their physical shape.
 */

export const CHUNK = 64;

export type Axis = 0 | 1 | 2;
export type Vec3 = [number, number, number];

export interface Plane {
	name: "xy" | "xz" | "yz";
	normal: Axis;
	u: Axis;
	v: Axis;
}

// Laid out 2 × 2 as XY | YZ over XZ, so rows share y and columns share x.
export const PLANES: Record<Plane["name"], Plane> = {
	xy: { name: "xy", normal: 0, u: 2, v: 1 },
	xz: { name: "xz", normal: 1, u: 2, v: 0 },
	yz: { name: "yz", normal: 2, u: 0, v: 1 },
};

export interface Level {
	index: number;
	/** The level's array, within the image. */
	path: string;
	/** Shape of the level, (z, y, x). */
	shape: Vec3;
	/** Level-0 voxels per voxel of this level, (z, y, x). */
	scale: Vec3;
}

export interface View {
	plane: Plane;
	position: Vec3;
	zoom: number;
	aspect: Vec3;
	width: number;
	height: number;
	/** Device pixels per CSS pixel; levels are chosen per CSS pixel. */
	pixelRatio?: number;
}

export interface TileKey {
	level: number;
	cz: number;
	cy: number;
	cx: number;
}

/** A rectangle of a plane: (u0, v0, u1, v1) in level-0 voxels, half-open. */
export type Rect = [number, number, number, number];

export function tileId(key: TileKey): string {
	return `${key.level}/${key.cz}/${key.cy}/${key.cx}`;
}

/**
 * Where a (z0, y0, x0, z1, y1, x1) box crosses the plane at level-0 index
 * `slice`, or null if it doesn't.
 */
export function boxOnPlane(box: readonly number[], plane: Plane, slice: number): Rect | null {
	const { normal, u, v } = plane;
	if (slice < box[normal]! || slice >= box[normal + 3]!) return null;
	return [box[u]!, box[v]!, box[u + 3]!, box[v + 3]!];
}

/** Whether a level-0 chunk's part of a plane overlaps `rect`. */
export function tileCrosses(key: TileKey, plane: Plane, rect: Rect): boolean {
	const corner = [key.cz, key.cy, key.cx].map((c) => c * CHUNK);
	const [u0, v0, u1, v1] = rect;
	const u = corner[plane.u]!;
	const v = corner[plane.v]!;
	return u < u1 && u + CHUNK > u0 && v < v1 && v + CHUNK > v0;
}

/** Physical voxel size per axis relative to the finest axis. */
export function aspectOf(voxelSize: Vec3 | null | undefined): Vec3 {
	if (!voxelSize || voxelSize.some((s) => !(s > 0))) return [1, 1, 1];
	const finest = Math.min(...voxelSize);
	return voxelSize.map((s) => s / finest) as Vec3;
}

/** Screen pixels per level-0 voxel along each axis. */
export function pixelsPerVoxel(view: View): Vec3 {
	return view.aspect.map((a) => a * view.zoom) as Vec3;
}

/** Device pixels a level's voxels cover on screen, along the plane's finer axis. */
function voxelPixels(level: Level, view: View): number {
	const { u, v } = view.plane;
	const px = pixelsPerVoxel(view);
	return Math.min(level.scale[u] * px[u], level.scale[v] * px[v]);
}

/**
 * Zooming in, a view keeps the level it shows until that level's voxels
 * cover this much more than the limit, so zooming back and forth near the
 * switch doesn't flip between two levels (reloading each time).
 */
export const LEVEL_HYSTERESIS = 1.25;

/**
 * The coarsest level whose voxels still cover at most about one and a half
 * CSS pixels in the view's plane. A view showing level `current` keeps it a
 * little past that point as it zooms in (see LEVEL_HYSTERESIS).
 */
export function chooseLevel(levels: Level[], view: View, current?: number): Level {
	let best = levels[0];
	if (!best) throw new Error("An image has at least one level");
	const limit = 1.5 * (view.pixelRatio ?? 1);
	for (const level of levels) {
		const size = voxelPixels(level, view);
		if (size <= limit && size > voxelPixels(best, view)) best = level;
	}
	const kept = current === undefined ? undefined : levels[current];
	if (
		kept &&
		kept.index > best.index &&
		voxelPixels(kept, view) > voxelPixels(best, view) &&
		voxelPixels(kept, view) <= limit * LEVEL_HYSTERESIS
	) {
		return kept;
	}
	return best;
}

/**
 * The level a view shows: the one `chooseLevel` picks, or a coarser one if
 * that would take more than `maxTiles` chunks to cover the view.
 */
export function viewLevel(levels: Level[], view: View, current: number | undefined, maxTiles: number): Level {
	let level = chooseLevel(levels, view, current);
	while (level.index < levels.length - 1 && countTiles(level, view) > maxTiles) level = levels[level.index + 1]!;
	return level;
}

/** The plane's index along its normal axis, in `level`'s voxels. */
export function sliceIndex(level: Level, view: View): number {
	const n = view.plane.normal;
	return Math.floor(Math.floor(view.position[n]) / level.scale[n]);
}

/**
 * The chunks of `level` a view shows, with `padding` more on every side, as
 * the chunk index along the normal and inclusive ranges along u and v (empty
 * when first > last); null if the plane misses the level.
 */
function tileRange(level: Level, view: View, padding: number) {
	const { normal, u, v } = view.plane;
	const slice = sliceIndex(level, view);
	if (slice < 0 || slice >= level.shape[normal]) return null;
	const px = pixelsPerVoxel(view);
	const span = (axis: Axis, half: number): [number, number] => {
		const center = view.position[axis] / level.scale[axis];
		const extent = half / (px[axis] * level.scale[axis]);
		const last = Math.ceil(level.shape[axis] / CHUNK) - 1;
		return [
			Math.max(0, Math.floor((center - extent) / CHUNK) - padding),
			Math.min(last, Math.floor((center + extent) / CHUNK) + padding),
		];
	};
	const [u0, u1] = span(u, view.width / 2);
	const [v0, v1] = span(v, view.height / 2);
	return { normal: Math.floor(slice / CHUNK), u0, u1, v0, v1 };
}

/** How many chunks of `level` the view shows (with `padding` more on every side). */
export function countTiles(level: Level, view: View, padding = 0): number {
	const range = tileRange(level, view, padding);
	if (!range) return 0;
	return Math.max(0, range.u1 - range.u0 + 1) * Math.max(0, range.v1 - range.v0 + 1);
}

/**
 * The chunks of `level` that the view shows, nearest its center on screen
 * first, then `padding` chunks around them (to load before they scroll into
 * view), also nearest first.
 */
export function visibleTiles(level: Level, view: View, padding = 1): TileKey[] {
	const inner = tileRange(level, view, 0);
	const outer = tileRange(level, view, padding);
	if (!inner || !outer) return [];
	const { normal, u, v } = view.plane;
	const px = pixelsPerVoxel(view);
	// Screen pixels from the view's center to a chunk's center along an axis.
	const offset = (axis: Axis, c: number) => ((c + 0.5) * CHUNK * level.scale[axis] - view.position[axis]) * px[axis];
	const tiles: (TileKey & { margin: boolean; distance: number })[] = [];
	for (let cv = outer.v0; cv <= outer.v1; cv++) {
		for (let cu = outer.u0; cu <= outer.u1; cu++) {
			const c: Vec3 = [0, 0, 0];
			c[normal] = outer.normal;
			c[u] = cu;
			c[v] = cv;
			const margin = cu < inner.u0 || cu > inner.u1 || cv < inner.v0 || cv > inner.v1;
			const distance = Math.hypot(offset(u, cu), offset(v, cv));
			tiles.push({ level: level.index, cz: c[0], cy: c[1], cx: c[2], margin, distance });
		}
	}
	tiles.sort((a, b) => Number(a.margin) - Number(b.margin) || a.distance - b.distance);
	return tiles.map(({ level, cz, cy, cx }) => ({ level, cz, cy, cx }));
}

/** Which of `keys`, chunks of `level`, the view shows, nearest its center first. */
export function tilesShown(keys: Iterable<TileKey>, level: Level, view: View): TileKey[] {
	const range = tileRange(level, view, 0);
	if (!range) return [];
	const { normal, u, v } = view.plane;
	const px = pixelsPerVoxel(view);
	const offset = (axis: Axis, c: number) => ((c + 0.5) * CHUNK * level.scale[axis] - view.position[axis]) * px[axis];
	const shown: (TileKey & { distance: number })[] = [];
	for (const key of keys) {
		const c = [key.cz, key.cy, key.cx];
		const [n, cu, cv] = [c[normal]!, c[u]!, c[v]!];
		if (n !== range.normal || cu < range.u0 || cu > range.u1 || cv < range.v0 || cv > range.v1) continue;
		shown.push({ ...key, distance: Math.hypot(offset(u, cu), offset(v, cv)) });
	}
	shown.sort((a, b) => a.distance - b.distance);
	return shown.map(({ level, cz, cy, cx }) => ({ level, cz, cy, cx }));
}

/**
 * The chunks a view loads, most wanted first: every level coarser than
 * `level`, coarsest first, where the view shows it (backdrops that arrive
 * quickly and show while finer chunks load), then `level` itself, nearest
 * the center first, then `padding` chunks around it.
 */
export function tilesToLoad(levels: Level[], level: Level, view: View, padding = 1): TileKey[] {
	const tiles: TileKey[] = [];
	for (let i = levels.length - 1; i > level.index; i--) tiles.push(...visibleTiles(levels[i]!, view, 0));
	tiles.push(...visibleTiles(level, view, padding));
	return tiles;
}

/**
 * The chunk of the coarser level `to` that holds `key`'s part of the plane
 * the view shows.
 */
export function coveringTile(key: TileKey, from: Level, to: Level, view: View): TileKey {
	const { normal, u, v } = view.plane;
	const c = [key.cz, key.cy, key.cx];
	const out: Vec3 = [0, 0, 0];
	out[normal] = Math.floor(sliceIndex(to, view) / CHUNK);
	for (const axis of [u, v]) out[axis] = Math.floor((c[axis]! * from.scale[axis]) / to.scale[axis]);
	return { level: to.index, cz: out[0], cy: out[1], cx: out[2] };
}

/** The level-0 voxel under a point on the view, from the view's center. */
export function voxelAt(view: View, dx: number, dy: number): Vec3 {
	const px = pixelsPerVoxel(view);
	const { u, v } = view.plane;
	const point = [...view.position] as Vec3;
	point[u] += dx / px[u];
	point[v] += dy / px[v];
	return point;
}

/** Map a stored value into [0, 1] for display, given a window [low, high]. */
export function windowed(value: number, low: number, high: number): number {
	if (high <= low) return value >= high ? 1 : 0;
	return Math.min(1, Math.max(0, (value - low) / (high - low)));
}
