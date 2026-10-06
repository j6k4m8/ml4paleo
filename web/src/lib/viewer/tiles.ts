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
}

export interface TileKey {
	level: number;
	cz: number;
	cy: number;
	cx: number;
}

export function tileId(key: TileKey): string {
	return `${key.level}/${key.cz}/${key.cy}/${key.cx}`;
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

/**
 * The coarsest level whose voxels still cover at most about one and a half
 * screen pixels in the view's plane.
 */
export function chooseLevel(levels: Level[], view: View): Level {
	let best = levels[0];
	if (!best) throw new Error("An image has at least one level");
	const { u, v } = view.plane;
	const px = pixelsPerVoxel(view);
	const size = (level: Level) => Math.min(level.scale[u] * px[u], level.scale[v] * px[v]);
	for (const level of levels) {
		if (size(level) <= 1.5 && size(level) > size(best)) best = level;
	}
	return best;
}

/** The plane's index along its normal axis, in `level`'s voxels. */
export function sliceIndex(level: Level, view: View): number {
	const n = view.plane.normal;
	return Math.floor(Math.floor(view.position[n]) / level.scale[n]);
}

/**
 * The chunks of `level` that the view shows (plus `padding` chunks around
 * it, to load before they scroll into view), nearest to the center first.
 */
export function visibleTiles(level: Level, view: View, padding = 1): TileKey[] {
	const { normal, u, v } = view.plane;
	const slice = sliceIndex(level, view);
	if (slice < 0 || slice >= level.shape[normal]) return [];
	const px = pixelsPerVoxel(view);
	const range = (axis: Axis, half: number) => {
		const center = view.position[axis] / level.scale[axis];
		const extent = half / (px[axis] * level.scale[axis]);
		const last = Math.ceil(level.shape[axis] / CHUNK) - 1;
		return {
			first: Math.max(0, Math.floor((center - extent) / CHUNK) - padding),
			last: Math.min(last, Math.floor((center + extent) / CHUNK) + padding),
			center: center / CHUNK - 0.5,
		};
	};
	const us = range(u, view.width / 2);
	const vs = range(v, view.height / 2);
	const tiles: (TileKey & { distance: number })[] = [];
	for (let cv = vs.first; cv <= vs.last; cv++) {
		for (let cu = us.first; cu <= us.last; cu++) {
			const c: Vec3 = [0, 0, 0];
			c[normal] = Math.floor(slice / CHUNK);
			c[u] = cu;
			c[v] = cv;
			const distance = Math.hypot(cu - us.center, cv - vs.center);
			tiles.push({ level: level.index, cz: c[0], cy: c[1], cx: c[2], distance });
		}
	}
	tiles.sort((a, b) => a.distance - b.distance);
	return tiles.map(({ level, cz, cy, cx }) => ({ level, cz, cy, cx }));
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
