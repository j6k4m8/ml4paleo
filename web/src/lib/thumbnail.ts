/**
 * Small pictures: of ROIs (the middle slice of the ROI from the image, at a
 * level near thumbnail size, with its labels when the ROI is small enough to
 * read them at full resolution), and of a whole image (a slice through its
 * middle).
 */

import * as zarr from "zarrita";
import { type Box, thinAxis } from "./rois.svelte";
import { absolute, shardedFetch } from "./viewer/image";
import { paletteBytes } from "./viewer/plane";
import { type Level, PLANES, type Plane } from "./viewer/tiles";
import { useZstd } from "./viewer/zstd";

const TARGET = 96;
const MAX_LABEL_EXTENT = 256;

useZstd();

type OpenArray = zarr.Array<zarr.DataType, zarr.FetchStore>;
const arrays = new Map<string, Promise<OpenArray>>();

/** Each array's metadata is read once for all thumbnails. */
function openArray(url: string, path: string): Promise<OpenArray> {
	const key = `${url}#${path}`;
	let found = arrays.get(key);
	if (!found) {
		const store = new zarr.FetchStore(absolute(url), { useSuffixRequest: true, fetch: shardedFetch() });
		found = zarr.open.v3(zarr.root(store).resolve(path), { kind: "array" });
		found.catch(() => arrays.delete(key));
		arrays.set(key, found);
	}
	return found;
}

/** The plane a thumbnail shows: the thin axis of a slice ROI, else XY. */
export function thumbnailPlane(bbox: Box): Plane {
	const normal = thinAxis(bbox);
	return normal === 0 ? PLANES.xy : normal === 1 ? PLANES.xz : PLANES.yz;
}

/**
 * The coarsest level that still shows the box at least `target` voxels along
 * its longer side (a thumbnail is square, so the shorter side shrinks).
 */
export function thumbnailLevel(levels: Level[], bbox: Box, plane: Plane, target = TARGET): Level {
	let best = levels[0]!;
	for (const level of levels) {
		const across = Math.max(
			(bbox[plane.u + 3]! - bbox[plane.u]!) / level.scale[plane.u],
			(bbox[plane.v + 3]! - bbox[plane.v]!) / level.scale[plane.v],
		);
		if (across >= target) best = level;
	}
	return best;
}

/** Read the plane through the ROI's middle at `level`, as rows (v) of columns (u). */
async function readPlane(
	url: string,
	path: string,
	channel: boolean,
	level: Pick<Level, "scale" | "shape">,
	bbox: Box,
	plane: Plane,
	signal?: AbortSignal,
): Promise<{ data: ArrayLike<number>; width: number; height: number }> {
	const array = await openArray(url, path);
	signal?.throwIfAborted();
	const selection: (number | zarr.Slice)[] = [];
	const lo = [0, 0, 0];
	const hi = [0, 0, 0];
	for (let axis = 0; axis < 3; axis++) {
		const scale = level.scale[axis]!;
		lo[axis] = Math.floor(bbox[axis]! / scale);
		hi[axis] = Math.min(level.shape[axis]!, Math.max(lo[axis]! + 1, Math.ceil(bbox[axis + 3]! / scale)));
	}
	const middle = Math.floor((lo[plane.normal]! + hi[plane.normal]! - 1) / 2);
	for (let axis = 0; axis < 3; axis++) {
		selection.push(axis === plane.normal ? middle : zarr.slice(lo[axis]!, hi[axis]!));
	}
	const result = await zarr.get(array, channel ? [0, ...selection] : selection, { opts: { signal } });
	const [rows = 1, cols = 1] = result.shape;
	const width = hi[plane.u]! - lo[plane.u]!;
	const height = hi[plane.v]! - lo[plane.v]!;
	const source = result.data as unknown as ArrayLike<number | bigint>;
	const out = new Float32Array(width * height);
	// The result's axes are the two in-plane axes in (z, y, x) order: rows
	// are v unless u comes first (the YZ plane).
	const transposed = plane.u < plane.v;
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const value = Number(source[r * cols + c]);
			if (transposed) out[c * width + r] = value;
			else out[r * width + c] = value;
		}
	}
	return { data: out, width, height };
}

/** Draw an ROI's thumbnail into `canvas`. */
export async function drawThumbnail(
	canvas: HTMLCanvasElement,
	options: {
		imageUrl: string;
		labelsUrl: string;
		levels: Level[];
		bbox: Box;
		window: [number, number];
		colors: Map<number, string>;
		opacity: number;
		signal?: AbortSignal;
	},
): Promise<void> {
	const { levels, bbox, signal } = options;
	const plane = thumbnailPlane(bbox);
	const level = thumbnailLevel(levels, bbox, plane);
	const image = await readPlane(options.imageUrl, level.path, true, level, bbox, plane, signal);
	const extent = Math.max(bbox[plane.u + 3]! - bbox[plane.u]!, bbox[plane.v + 3]! - bbox[plane.v]!);
	const labels =
		level.index === 0 && extent <= MAX_LABEL_EXTENT
			? await readPlane(options.labelsUrl, "class", false, { scale: [1, 1, 1], shape: levels[0]!.shape }, bbox, plane, signal)
			: null;
	paint(canvas, image, options.window, labels, paletteBytes(options.colors), options.opacity);
}

/** Draw a slice through the middle of the whole image into `canvas`, at a level near `target` pixels across. */
export async function drawImageSlice(
	canvas: HTMLCanvasElement,
	options: {
		imageUrl: string;
		levels: Level[];
		plane: Plane;
		window: [number, number];
		target?: number;
		signal?: AbortSignal;
	},
): Promise<void> {
	const { levels, plane, signal } = options;
	const [z = 1, y = 1, x = 1] = levels[0]!.shape;
	const bbox: Box = [0, 0, 0, z, y, x];
	const level = thumbnailLevel(levels, bbox, plane, options.target);
	const image = await readPlane(options.imageUrl, level.path, true, level, bbox, plane, signal);
	signal?.throwIfAborted();
	paint(canvas, image, options.window, null, null, 0);
}

/** Put a windowed grayscale slice, with labels blended in when given, into `canvas`. */
function paint(
	canvas: HTMLCanvasElement,
	image: { data: ArrayLike<number>; width: number; height: number },
	window: [number, number],
	labels: { data: ArrayLike<number> } | null,
	palette: Uint8Array | null,
	opacity: number,
): void {
	canvas.width = image.width;
	canvas.height = image.height;
	const context = canvas.getContext("2d");
	if (!context) return;
	const pixels = context.createImageData(image.width, image.height);
	const [low, high] = window;
	for (let i = 0; i < image.width * image.height; i++) {
		const shade = Math.round(255 * Math.min(1, Math.max(0, (image.data[i]! - low) / (high - low || 1))));
		let [r, g, b] = [shade, shade, shade];
		const value = labels ? labels.data[i]! : 0;
		if (palette && value > 0 && palette[value * 4 + 3]) {
			r = Math.round(r * (1 - opacity) + palette[value * 4]! * opacity);
			g = Math.round(g * (1 - opacity) + palette[value * 4 + 1]! * opacity);
			b = Math.round(b * (1 - opacity) + palette[value * 4 + 2]! * opacity);
		}
		pixels.data.set([r, g, b, 255], i * 4);
	}
	context.putImageData(pixels, 0, 0);
}
