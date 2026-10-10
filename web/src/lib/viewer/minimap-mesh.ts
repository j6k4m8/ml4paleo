/** zmesh preview wire format, triangle styling, depth sorting and exact surface picking. */
import type { Chunk } from "./chunks";
import { classOpacity, type ClassStyles } from "./class-display";
import { labelKey } from "./labels";
import { CHUNK, type Level, type Vec3 } from "./tiles";

// Three world-space vertices, face normal, rgba, suggestion flag (one instance per triangle).
export const MESH_STRIDE = 17;
export const MAX_MESH_TRIANGLES = 120_000;
export interface SurfaceInput {
	level: Level;
	chunks: readonly { id: string; chunk: Chunk }[];
	colors: ReadonlyMap<number, string>;
	shape: Vec3;
	aspect: Vec3;
	styles?: ClassStyles;
}

/** Pack only the bounded resident snapshot; this never posts annotation edits. */
export function encodePreview(input: SurfaceInput): ArrayBuffer {
	const chunks = input.chunks.map(({ id, chunk }) => {
		const key = labelKey(id), origin = [key.cz * CHUNK, key.cy * CHUNK, key.cx * CHUNK];
		const shape = chunk.shape.map((n, i) => Math.max(0, Math.min(n, input.level.shape[i]! - origin[i]!)));
		return { chunk, origin, shape };
	}).filter(({ shape }) => shape.every((n) => n > 0));
	if (chunks.length > 16) throw new Error("Too many 3D preview chunks.");
	const size = 16 + chunks.reduce((n, c) => n + 24 + c.shape.reduce((a, b) => a * b, 1), 0);
	const body = new ArrayBuffer(size), header = new DataView(body), bytes = new Uint8Array(body);
	let offset = 0;
	const ints = (values: readonly number[]) => { for (const n of values) { header.setUint32(offset, n, true); offset += 4; } };
	ints([...input.level.shape, chunks.length]);
	const visible = Uint8Array.from({ length: 256 }, (_, v) => v > 1 && v < 255 && classOpacity(v, input.styles ?? {}) > 0 ? v : 0);
	for (const { chunk, origin, shape } of chunks) {
		ints([...origin, ...shape]);
		const data = chunk.data as Uint8Array;
		for (let z = 0; z < shape[0]!; z++) for (let y = 0; y < shape[1]!; y++) for (let x = 0; x < shape[2]!; x++) {
			bytes[offset++] = visible[data[(z * chunk.shape[1]! + y) * chunk.shape[2]! + x]!]!;
		}
	}
	return body;
}

/** Native xyz coarse-grid vertices become physical xyz world coordinates, preserving partial edge cells. */
export function decodePreview(buffer: ArrayBuffer, input: SurfaceInput, suggested = false): Float32Array {
	if (buffer.byteLength % 40 || buffer.byteLength / 40 > MAX_MESH_TRIANGLES) throw new Error("Invalid 3D surface response.");
	const wire = new DataView(buffer), count = buffer.byteLength / 40, out = new Float32Array(count * MESH_STRIDE);
	const longest = Math.max(...input.shape.map((n, i) => n * input.aspect[i]!));
	for (let i = 0; i < count; i++) {
		const value = wire.getFloat32(i * 40, true);
		if (!Number.isInteger(value) || value < 2 || value >= 255) throw new Error("Invalid 3D surface class.");
		const points: number[] = [];
		for (let v = 0; v < 3; v++) for (let axis = 0; axis < 3; axis++) {
			const a = 2 - axis, coordinate = wire.getFloat32(i * 40 + 4 * (1 + v * 3 + axis), true);
			if (!Number.isFinite(coordinate)) throw new Error("Invalid 3D surface coordinate.");
			const scale = input.level.scale[a]!, cells = input.level.shape[a]!, full = input.shape[a]!;
			// Interpolate within the last partial cell instead of collapsing its surface.
			const voxel = coordinate > cells - 1 ? (cells - 1) * scale + (coordinate - cells + 1) * (full - (cells - 1) * scale) : coordinate * scale;
			points.push((Math.max(0, Math.min(full, voxel)) - full / 2) * input.aspect[a]! / longest * (axis === 1 ? -1 : 1));
		}
		const u = points.slice(3, 6).map((n, a) => n - points[a]!);
		const v = points.slice(6, 9).map((n, a) => n - points[a]!);
		const normal = [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
		const length = Math.hypot(...normal) || 1;
		const hex = Number.parseInt((input.colors.get(value) ?? "#8aa0b8").slice(1), 16);
		out.set([...points, ...normal.map((n) => n / length), (hex >> 16 & 255) / 255, (hex >> 8 & 255) / 255, (hex & 255) / 255,
			0.78 * classOpacity(value, input.styles ?? {}), Number(suggested)], i * MESH_STRIDE);
	}
	return out;
}

export interface MeshRay { origin: Vec3; direction: Vec3 }
/** Two-sided Möller–Trumbore intersection of the exact rendered triangles. */
export function intersectMesh(mesh: Float32Array, ray: MeshRay, nearest = Infinity): { distance: number; point: Vec3 } | null {
	let hit = false;
	for (let i = 0; i < mesh.length; i += MESH_STRIDE) {
		if (mesh[i + 15]! <= 0) continue;
		const ax = mesh[i]!, ay = mesh[i + 1]!, az = mesh[i + 2]!;
		const ux = mesh[i + 3]! - ax, uy = mesh[i + 4]! - ay, uz = mesh[i + 5]! - az;
		const vx = mesh[i + 6]! - ax, vy = mesh[i + 7]! - ay, vz = mesh[i + 8]! - az;
		const [dx, dy, dz] = ray.direction;
		const px = dy * vz - dz * vy, py = dz * vx - dx * vz, pz = dx * vy - dy * vx;
		const det = ux * px + uy * py + uz * pz;
		if (Math.abs(det) < 1e-12) continue;
		const tx = ray.origin[0] - ax, ty = ray.origin[1] - ay, tz = ray.origin[2] - az;
		const b = (tx * px + ty * py + tz * pz) / det;
		if (b < -1e-7 || b > 1 + 1e-7) continue;
		const qx = ty * uz - tz * uy, qy = tz * ux - tx * uz, qz = tx * uy - ty * ux;
		const c = (dx * qx + dy * qy + dz * qz) / det;
		if (c < -1e-7 || b + c > 1 + 1e-7) continue;
		const t = (vx * qx + vy * qy + vz * qz) / det;
		if (t < 0.02 || t >= nearest || t > 20) continue;
		nearest = t; hit = true;
	}
	return hit ? { distance: nearest, point: ray.origin.map((n, a) => n + nearest * ray.direction[a]!) as Vec3 } : null;
}

/** One back-to-front ordering for saved and proposed triangles, cached between orbits. */
export function sortMesh(mesh: Float32Array, back: Vec3): Float32Array {
	const n = mesh.length / MESH_STRIDE, depths = new Float32Array(n);
	for (let i = 0; i < n; i++) for (let v = 0; v < 3; v++) {
		const a = i * MESH_STRIDE + v * 3;
		depths[i] = depths[i]! + (mesh[a]! * back[0] + mesh[a + 1]! * back[1] + mesh[a + 2]! * back[2]) / 3;
	}
	const order = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => depths[a]! - depths[b]!);
	const sorted = new Float32Array(mesh.length);
	for (let i = 0; i < n; i++) sorted.set(mesh.subarray(order[i]! * MESH_STRIDE, (order[i]! + 1) * MESH_STRIDE), i * MESH_STRIDE);
	return sorted;
}
