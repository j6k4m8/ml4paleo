/**
 * Which pyramid level and which chunks a plane view needs.
 *
 * Coordinates are in full-resolution (level 0) voxels, ordered (z, y, x) as
 * in storage. A view looks at one z plane; `zoom` is screen pixels per
 * level-0 voxel.
 */

export const CHUNK = 64;

export interface Level {
	index: number;
	/** Shape of the level, (z, y, x). */
	shape: [number, number, number];
	/** Level-0 voxels per voxel of this level, (z, y, x). */
	scale: [number, number, number];
}

export interface View {
	z: number;
	centerY: number;
	centerX: number;
	zoom: number;
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

/**
 * The coarsest level whose voxels still cover at most about one screen pixel.
 */
export function chooseLevel(levels: Level[], zoom: number): Level {
	let best = levels[0];
	if (!best) throw new Error("An image has at least one level");
	for (const level of levels) {
		const inPlane = Math.min(level.scale[1], level.scale[2]);
		if (inPlane * zoom <= 1.5 && inPlane > Math.min(best.scale[1], best.scale[2])) best = level;
	}
	return best;
}

/**
 * The chunks of `level` that the view shows (plus `padding` chunks around
 * it, to load before they scroll into view), nearest to the center first.
 */
export function visibleTiles(level: Level, view: View, padding = 1): TileKey[] {
	const [sz, sy, sx] = level.scale;
	const [nz, ny, nx] = level.shape;
	const z = Math.floor(view.z / sz);
	if (z < 0 || z >= nz) return [];
	const halfH = view.height / 2 / view.zoom;
	const halfW = view.width / 2 / view.zoom;
	const lastY = Math.ceil(ny / CHUNK) - 1;
	const lastX = Math.ceil(nx / CHUNK) - 1;
	const y0 = Math.max(0, Math.floor((view.centerY - halfH) / sy / CHUNK) - padding);
	const y1 = Math.min(lastY, Math.floor((view.centerY + halfH) / sy / CHUNK) + padding);
	const x0 = Math.max(0, Math.floor((view.centerX - halfW) / sx / CHUNK) - padding);
	const x1 = Math.min(lastX, Math.floor((view.centerX + halfW) / sx / CHUNK) + padding);
	const cz = Math.floor(z / CHUNK);
	const tiles: TileKey[] = [];
	for (let cy = y0; cy <= y1; cy++) {
		for (let cx = x0; cx <= x1; cx++) tiles.push({ level: level.index, cz, cy, cx });
	}
	const centerCy = view.centerY / sy / CHUNK - 0.5;
	const centerCx = view.centerX / sx / CHUNK - 0.5;
	tiles.sort(
		(a, b) =>
			Math.hypot(a.cy - centerCy, a.cx - centerCx) - Math.hypot(b.cy - centerCy, b.cx - centerCx),
	);
	return tiles;
}

/** Map a stored value into [0, 1] for display, given a window [low, high]. */
export function windowed(value: number, low: number, high: number): number {
	if (high <= low) return value >= high ? 1 : 0;
	return Math.min(1, Math.max(0, (value - low) / (high - low)));
}
