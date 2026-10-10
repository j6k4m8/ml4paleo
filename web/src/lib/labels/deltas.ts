/**
 * Label edits as per-chunk mask deltas, the browser side of
 * `ml4paleo.labels.deltas`: the same bit packing, boxes, and split, checked
 * against the shared fixtures in `tests/fixtures/labels/cases.json`.
 *
 * Masks are 0/1 bytes in C order (z, y, x). Payloads are zstd frames made of
 * raw and run-length blocks (no entropy coding), which any zstd decoder
 * reads and which are never more than a few bytes larger than their
 * contents, so they stay inside the server's decompression bounds.
 */

import { decompress } from "fzstd";
import type { Vec3 } from "../viewer/tiles";

export const CHUNK = 64;

/** One chunk's part of an edit, as the API takes it. */
export interface DeltaIn {
	key: Vec3;
	base_version: number;
	/** Chunk-local (z0, y0, x0, z1, y1, x1), half-open. */
	box: [number, number, number, number, number, number];
	/** base64 zstd frame of the packed mask. */
	mask: string;
	value?: number;
	values?: string;
	only_if: string;
	/** For a decline: the model value the server must find under the tombstone. */
	prediction_value?: number;
}

/** Pack 0/1 bytes into bits, least significant bit first (numpy's "little"). */
export function packBits(mask: Uint8Array): Uint8Array {
	const out = new Uint8Array((mask.length + 7) >> 3);
	for (let i = 0; i < mask.length; i++) if (mask[i]) out[i >> 3]! |= 1 << (i & 7);
	return out;
}

export function unpackBits(bits: Uint8Array, count: number): Uint8Array {
	const out = new Uint8Array(count);
	for (let i = 0; i < count; i++) out[i] = (bits[i >> 3]! >> (i & 7)) & 1;
	return out;
}

const MAX_BLOCK = 128 * 1024;
// Shorter runs cost more as their own block than they save.
const MIN_RUN = 16;

/** A zstd frame holding `data`, with run-length blocks for long runs. */
export function zstdFrame(data: Uint8Array): Uint8Array {
	const parts: number[] = [0x28, 0xb5, 0x2f, 0xfd];
	const size = data.length;
	// Single segment, so the content size is the window: no window byte.
	if (size < 256) parts.push(0x20, size);
	else if (size < 65536 + 256) parts.push(0x60, (size - 256) & 255, (size - 256) >> 8);
	else parts.push(0xa0, size & 255, (size >> 8) & 255, (size >> 16) & 255, size >>> 24);

	const blocks: { type: 0 | 1; start: number; length: number }[] = [];
	const raw = (start: number, end: number) => {
		for (let at = start; at < end; at += MAX_BLOCK) {
			blocks.push({ type: 0, start: at, length: Math.min(MAX_BLOCK, end - at) });
		}
	};
	let rawStart = 0;
	let i = 0;
	while (i < size) {
		let j = i + 1;
		while (j < size && data[j] === data[i]) j++;
		if (j - i >= MIN_RUN) {
			raw(rawStart, i);
			for (let at = i; at < j; at += MAX_BLOCK) {
				blocks.push({ type: 1, start: at, length: Math.min(MAX_BLOCK, j - at) });
			}
			rawStart = j;
		}
		i = j;
	}
	raw(rawStart, size);
	if (blocks.length === 0) blocks.push({ type: 0, start: 0, length: 0 });

	const header = (block: (typeof blocks)[number], last: boolean) => {
		const value = (block.length << 3) | (block.type << 1) | (last ? 1 : 0);
		return [value & 255, (value >> 8) & 255, (value >> 16) & 255];
	};
	let length = parts.length;
	for (const block of blocks) length += 3 + (block.type === 0 ? block.length : 1);
	const out = new Uint8Array(length);
	out.set(parts);
	let at = parts.length;
	blocks.forEach((block, index) => {
		out.set(header(block, index === blocks.length - 1), at);
		at += 3;
		if (block.type === 0) {
			out.set(data.subarray(block.start, block.start + block.length), at);
			at += block.length;
		} else {
			out[at++] = data[block.start]!;
		}
	});
	return out;
}

export function base64(bytes: Uint8Array): string {
	let text = "";
	for (let i = 0; i < bytes.length; i += 0x8000) {
		text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}
	return btoa(text);
}

export function fromBase64(text: string): Uint8Array {
	return Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

export interface SplitOptions {
	value?: number;
	/** Per-voxel values, shaped like the mask. */
	values?: Uint8Array;
	onlyIf?: string;
	/** Chunk versions the client saw, by `cz/cy/cx`, for strict edits. */
	baseVersions?: Map<string, number>;
}

/**
 * Split a mask of `shape` placed at `origin` (level-0 voxels) into one delta
 * per chunk it touches, each cropped to the tight box around its voxels.
 */
export function splitIntoDeltas(mask: Uint8Array, shape: Vec3, origin: Vec3, options: SplitOptions): DeltaIn[] {
	if ((options.value === undefined) === (options.values === undefined)) {
		throw new Error("Pass exactly one of value or values");
	}
	const [sz, sy, sx] = shape;
	const lo: Vec3 = [Infinity, Infinity, Infinity];
	const hi: Vec3 = [-1, -1, -1];
	for (let z = 0; z < sz; z++) {
		for (let y = 0; y < sy; y++) {
			const row = (z * sy + y) * sx;
			for (let x = 0; x < sx; x++) {
				if (!mask[row + x]) continue;
				lo[0] = Math.min(lo[0], z);
				lo[1] = Math.min(lo[1], y);
				lo[2] = Math.min(lo[2], x);
				hi[0] = Math.max(hi[0], z);
				hi[1] = Math.max(hi[1], y);
				hi[2] = Math.max(hi[2], x);
			}
		}
	}
	if (hi[0] < 0) return [];
	const first = lo.map((l, a) => Math.floor((origin[a]! + l) / CHUNK));
	const last = hi.map((h, a) => Math.floor((origin[a]! + h) / CHUNK));
	const deltas: DeltaIn[] = [];
	for (let cz = first[0]!; cz <= last[0]!; cz++) {
		for (let cy = first[1]!; cy <= last[1]!; cy++) {
			for (let cx = first[2]!; cx <= last[2]!; cx++) {
				const key: Vec3 = [cz, cy, cx];
				const delta = chunkDelta(mask, shape, origin, key, options);
				if (delta) deltas.push(delta);
			}
		}
	}
	return deltas;
}

function chunkDelta(mask: Uint8Array, shape: Vec3, origin: Vec3, key: Vec3, options: SplitOptions): DeltaIn | null {
	// The part of the mask inside this chunk, in mask coordinates.
	const from = key.map((k, a) => Math.max(k * CHUNK - origin[a]!, 0));
	const to = key.map((k, a) => Math.min((k + 1) * CHUNK - origin[a]!, shape[a]!));
	if (from.some((f, a) => f >= to[a]!)) return null;
	const [, sy, sx] = shape;
	const lo = [Infinity, Infinity, Infinity];
	const hi = [-1, -1, -1];
	for (let z = from[0]!; z < to[0]!; z++) {
		for (let y = from[1]!; y < to[1]!; y++) {
			for (let x = from[2]!; x < to[2]!; x++) {
				if (!mask[(z * sy + y) * sx + x]) continue;
				const p = [z, y, x];
				for (let a = 0; a < 3; a++) {
					lo[a] = Math.min(lo[a]!, p[a]!);
					hi[a] = Math.max(hi[a]!, p[a]!);
				}
			}
		}
	}
	if (hi[0]! < 0) return null;
	const size = lo.map((l, a) => hi[a]! + 1 - l);
	const count = size[0]! * size[1]! * size[2]!;
	const part = new Uint8Array(count);
	const partValues = options.values ? new Uint8Array(count) : null;
	let i = 0;
	for (let z = lo[0]!; z <= hi[0]!; z++) {
		for (let y = lo[1]!; y <= hi[1]!; y++) {
			for (let x = lo[2]!; x <= hi[2]!; x++, i++) {
				const at = (z * sy + y) * sx + x;
				part[i] = mask[at] ? 1 : 0;
				if (partValues && options.values) partValues[i] = part[i] ? options.values[at]! : 0;
			}
		}
	}
	const local = lo.map((l, a) => origin[a]! + l - key[a]! * CHUNK);
	const delta: DeltaIn = {
		key,
		base_version: options.baseVersions?.get(key.join("/")) ?? 0,
		box: [local[0]!, local[1]!, local[2]!, local[0]! + size[0]!, local[1]! + size[1]!, local[2]! + size[2]!],
		mask: base64(zstdFrame(packBits(part))),
		only_if: options.onlyIf ?? "any",
	};
	if (partValues) delta.values = base64(zstdFrame(partValues));
	else delta.value = options.value;
	return delta;
}

/** A delta's mask (0/1 per voxel of its box) and what it writes. */
export function decodeDelta(delta: DeltaIn): { mask: Uint8Array; written: Uint8Array | number } {
	const [z0, y0, x0, z1, y1, x1] = delta.box;
	const count = (z1 - z0) * (y1 - y0) * (x1 - x0);
	const mask = unpackBits(decompress(fromBase64(delta.mask)), count);
	const written = delta.values !== undefined ? decompress(fromBase64(delta.values)) : (delta.value ?? 0);
	return { mask, written };
}

/**
 * Which values a voxel may hold for an `only_if` condition to let a delta
 * change it, as `ml4paleo.labels.deltas` judges it: 1 at each such value.
 * "any" allows every value; "unlabeled" only 0; "labeled" every value but 0
 * (background, 1, included); "class:2,3" the values it lists. A condition
 * this doesn't know allows every value, as it did before there were lists.
 */
export function allowedValues(onlyIf: string): Uint8Array {
	const allowed = new Uint8Array(256);
	if (onlyIf === "unlabeled") {
		allowed[0] = 1;
	} else if (onlyIf === "labeled") {
		allowed.fill(1, 1);
	} else if (onlyIf.startsWith("class:")) {
		const listed = onlyIf.slice("class:".length).split(",");
		if (listed.every((value) => /^\d{1,3}$/.test(value) && Number(value) < 256)) {
			for (const value of listed) allowed[Number(value)] = 1;
		} else {
			allowed.fill(1);
		}
	} else {
		allowed.fill(1);
	}
	return allowed;
}

/**
 * Apply a delta to a chunk's class values in place, as the server will
 * (`apply_delta`), so an edit shows before the server confirms it.
 * `chunkShape` may be smaller than 64³ at the volume's edges.
 */
export function applyLocally(
	chunk: Uint8Array,
	chunkShape: number[],
	box: DeltaIn["box"],
	mask: Uint8Array,
	written: Uint8Array | number,
	onlyIf = "any",
): number {
	const [, cy = 0, cx = 0] = chunkShape;
	const allowed = allowedValues(onlyIf);
	let changed = 0;
	let i = 0;
	for (let z = box[0]; z < box[3]; z++) {
		for (let y = box[1]; y < box[4]; y++) {
			for (let x = box[2]; x < box[5]; x++, i++) {
				if (!mask[i] || z >= (chunkShape[0] ?? 0) || y >= cy || x >= cx) continue;
				const at = (z * cy + y) * cx + x;
				if (!allowed[chunk[at]!]) continue;
				const value = typeof written === "number" ? written : written[i]!;
				if (chunk[at] !== value) changed++;
				chunk[at] = value;
			}
		}
	}
	return changed;
}
