/**
 * The annotator's small 3D overview. It deliberately uses plain WebGL2:
 * coarse live labels are zmesh surfaces, while the volume box, current orthogonal
 * slices, and crosshair provide spatial context. Coordinates preserve the
 * scan's physical voxel proportions.
 */

import type { Chunk } from "./chunks";
import { intersectMesh, sortMesh, MESH_STRIDE, type MeshRay } from "./minimap-mesh";
import type { Box } from "../rois.svelte";
import { intersection, liveKeys } from "./live";
import { CHUNK, type Level, type Vec3 } from "./tiles";

export const MAX_MINIMAP_VOXELS = 4 * 1024 * 1024;
export const MAX_MINIMAP_CHUNKS = 16;
export const MAX_MINIMAP_SIDE = 256;

export interface MinimapChunk {
	id: string;
	chunk: Chunk;
}

export interface MinimapSuggestionArea {
	box: Box;
	/** Live previews only expose already-computed chunks; never ask for more inference here. */
	keys?: readonly string[];
}

/** A bounded full-resolution neighborhood, even when a whole-volume prediction exists. */
export function minimapSuggestionIds(shape: Vec3, position: Vec3, sources: readonly MinimapSuggestionArea[], limit = MAX_MINIMAP_CHUNKS): string[] {
	const ids = new Set<string>();
	for (const source of sources) {
		const box = intersection(source.box, [0, 0, 0, ...shape]);
		if (!box) continue;
		const center = position.map((n, i) => Math.max(box[i]!, Math.min(box[i + 3]! - 1, n))) as Vec3;
		for (const id of source.keys ?? liveKeys(shape, center, [], null, 729)) {
			const key = id.split("/").map(Number);
			if (key.length !== 3 || key.some((n, i) => !Number.isInteger(n) || n < 0 || n * CHUNK >= shape[i]!)) continue;
			const lo = key.map((n) => n * CHUNK);
			if (intersection(box, [...lo, ...lo.map((n, i) => Math.min(n + CHUNK, shape[i]!))] as Box)) ids.add(id);
		}
	}
	const distance = (id: string) => id.split("/").reduce((sum, n, i) => sum + (Number(n) - Math.floor(position[i]! / CHUNK)) ** 2, 0);
	return [...ids].sort((a, b) => distance(a) - distance(b) || a.localeCompare(b)).slice(0, Math.max(0, limit));
}

/**
 * Suggestions are full-resolution and masked by ALL saved label state, even
 * hidden labels and declined/background voxels. Missing masks fail closed.
 * Sources are newest first: a proposal owns its box even before it has loaded.
 */
export function suggestionChunks(
	ids: readonly string[], sources: readonly { box: Box; chunks: ReadonlyMap<string, Chunk> }[],
	labels: ReadonlyMap<string, Chunk>, colors: ReadonlyMap<number, string>,
): MinimapChunk[] {
	const chunks: MinimapChunk[] = [];
	for (const id of ids) {
		const mask = labels.get(id);
		if (!mask) continue;
		const [nz, ny, nx] = mask.shape as Vec3;
		const origin = id.split("/").map((n) => Number(n) * CHUNK);
		const localSources = sources.map(({ box, chunks }) => ({ box, chunk: chunks.get(id) }));
		const data = new Uint8Array(nz * ny * nx);
		const saved = mask.data as Uint8Array;
		let index = 0;
		for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++, index++) {
			if (saved[index] !== 0) continue;
			const gz = origin[0]! + z, gy = origin[1]! + y, gx = origin[2]! + x;
			for (const { box, chunk } of localSources) {
				if (gz < box[0] || gy < box[1] || gx < box[2] || gz >= box[3] || gy >= box[4] || gx >= box[5]) continue;
				if (chunk && z < chunk.shape[0]! && y < chunk.shape[1]! && x < chunk.shape[2]!) {
					const value = (chunk.data as Uint8Array)[(z * chunk.shape[1]! + y) * chunk.shape[2]! + x]!;
					if (value > 1 && value < 255 && colors.has(value)) data[index] = value;
				}
				break;
			}
		}
		chunks.push({ id, chunk: { data, shape: mask.shape } });
	}
	return chunks;
}

/** The finest coarse label level small enough to keep resident for 3D. */
export function minimapLevel(levels: readonly Level[]): Level | null {
	if (levels.length === 0) return null;
	for (const level of levels) {
		const voxels = level.shape.reduce((a, b) => a * b, 1);
		const chunks = level.shape.reduce((a, b) => a * Math.ceil(b / CHUNK), 1);
		if (voxels <= MAX_MINIMAP_VOXELS && chunks <= MAX_MINIMAP_CHUNKS && Math.max(...level.shape) <= MAX_MINIMAP_SIDE) return level;
	}
	return levels.at(-1) ?? null;
}

/** Every chunk id of one label level, in storage order. */
export function minimapIds(level: Level): string[] {
	const ids: string[] = [];
	for (let z = 0; z < Math.ceil(level.shape[0] / CHUNK); z++) {
		for (let y = 0; y < Math.ceil(level.shape[1] / CHUNK); y++) {
			for (let x = 0; x < Math.ceil(level.shape[2] / CHUNK); x++) {
				ids.push(level.index === 0 ? `${z}/${y}/${x}` : `${level.index}/${z}/${y}/${x}`);
			}
		}
	}
	return ids;
}

/** Physical half-extents in the renderer's x,y,z order, normalized to fit. */
export function worldHalfExtents(shape: Vec3, aspect: Vec3): Vec3 {
	const physical = shape.map((n, axis) => n * aspect[axis]!) as Vec3;
	const longest = Math.max(...physical) || 1;
	return [physical[2] / longest / 2, physical[1] / longest / 2, physical[0] / longest / 2];
}

/** A level-0 z,y,x point mapped into the renderer's x,y,z world. */
export function worldPoint(point: Vec3, shape: Vec3, aspect: Vec3): Vec3 {
	const half = worldHalfExtents(shape, aspect);
	return [
		(point[2] / shape[2] - 0.5) * 2 * half[0],
		(0.5 - point[1] / shape[1]) * 2 * half[1],
		(point[0] / shape[0] - 0.5) * 2 * half[2],
	];
}

type Mat4 = Float32Array;
const HALF_FOV = Math.PI / 8;

export interface MinimapCamera {
	yaw: number;
	pitch: number;
	distance: number;
}

export function cameraBasis({ yaw, pitch }: MinimapCamera): { right: Vec3; up: Vec3; back: Vec3 } {
	const sy = Math.sin(yaw), cy = Math.cos(yaw), sp = Math.sin(pitch), cp = Math.cos(pitch);
	return { right: [cy, 0, -sy], up: [-sp * sy, cp, -sp * cy], back: [cp * sy, sp, cp * cy] };
}

/** Grab-style screen-plane pan in CSS pixels, returned in shared level-0 z,y,x coordinates. */
export function minimapPan(position: Vec3, dx: number, dy: number, height: number, camera: MinimapCamera, shape: Vec3, aspect: Vec3): Vec3 {
	const { right, up } = cameraBasis(camera);
	const perPixel = 2 * camera.distance * Math.tan(HALF_FOV) / Math.max(1, height);
	const delta = right.map((r, i) => (-dx * r + dy * up[i]!) * perPixel);
	const longest = Math.max(...shape.map((n, i) => n * aspect[i]!));
	return [
		position[0] + delta[2]! * longest / aspect[0],
		position[1] - delta[1]! * longest / aspect[1],
		position[2] + delta[0]! * longest / aspect[2],
	];
}

/** Orbit around the shared slice center, not the volume's fixed origin. */
export function minimapTransform(position: Vec3, camera: MinimapCamera, shape: Vec3, aspect: Vec3, width: number, height: number): Mat4 {
	const target = worldPoint(position, shape, aspect);
	const { right: x, up: y, back: z } = cameraBasis(camera);
	const eye = target.map((n, i) => n + camera.distance * z[i]!) as Vec3;
	const view = new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
	return multiply(perspective(width / Math.max(1, height)), view);
}

function multiply(a: Mat4, b: Mat4): Mat4 {
	const out = new Float32Array(16);
	for (let col = 0; col < 4; col++) {
		for (let row = 0; row < 4; row++) {
			let value = 0;
			for (let k = 0; k < 4; k++) value += a[k * 4 + row]! * b[col * 4 + k]!;
			out[col * 4 + row] = value;
		}
	}
	return out;
}

function perspective(aspect: number): Mat4 {
	const near = 0.02;
	const far = 20;
	const f = 1 / Math.tan(HALF_FOV);
	return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0]);
}

const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function shader(gl: WebGL2RenderingContext, kind: number, source: string): WebGLShader {
	const made = gl.createShader(kind);
	if (!made) throw new Error("Couldn't create a 3D preview shader");
	gl.shaderSource(made, source);
	gl.compileShader(made);
	if (!gl.getShaderParameter(made, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(made) ?? "Couldn't compile a 3D preview shader");
	return made;
}

function program(gl: WebGL2RenderingContext, mesh = false): WebGLProgram {
	const made = gl.createProgram();
	if (!made) throw new Error("Couldn't create the 3D preview");
	gl.attachShader(made, shader(gl, gl.VERTEX_SHADER, mesh ? `#version 300 es
		layout(location=0) in vec3 a;
		layout(location=1) in vec3 b;
		layout(location=2) in vec3 c;
		layout(location=3) in vec3 normal;
		layout(location=4) in vec4 color;
		layout(location=5) in float suggested;
		uniform mat4 transform;
		uniform vec2 opacity;
		out vec4 tint;
		flat out float striped;
		void main() {
			vec3 point = gl_VertexID == 0 ? a : gl_VertexID == 1 ? b : c;
			gl_Position = transform * vec4(point, 1.0);
			float light = 0.72 + 0.28 * abs(dot(normal, normalize(vec3(0.4, 0.8, 0.5))));
			tint = vec4(color.rgb * light, color.a * (suggested > 0.5 ? opacity.y : opacity.x));
			striped = suggested;
		}` : `#version 300 es
		layout(location=0) in vec3 position;
		layout(location=1) in vec4 color;
		uniform mat4 transform;
		out vec4 tint;
		void main() { gl_Position = transform * vec4(position, 1.0); tint = color; }`));
	gl.attachShader(made, shader(gl, gl.FRAGMENT_SHADER, mesh ? `#version 300 es
		precision highp float;
		in vec4 tint;
		flat in float striped;
		uniform float pixelRatio;
		out vec4 color;
		void main() {
			if (tint.a <= 0.0) discard;
			// Hatching changes the surface tint, not its geometry or picking.
			float hatch = striped > 0.5 && mod((gl_FragCoord.x + gl_FragCoord.y) / pixelRatio, 7.0) < 2.5 ? 0.35 : 1.0;
			color = vec4(tint.rgb * hatch, tint.a);
		}` : `#version 300 es
		precision mediump float;
		in vec4 tint;
		out vec4 color;
		void main() { color = tint; }`));
	gl.linkProgram(made);
	if (!gl.getProgramParameter(made, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(made) ?? "Couldn't link the 3D preview");
	return made;
}

function vertex(target: number[], point: Vec3, color: [number, number, number, number]) {
	target.push(...point, ...color);
}

function scene(shape: Vec3, aspect: Vec3, position: Vec3): { planes: Float32Array; lines: Float32Array } {
	const [hx, hy, hz] = worldHalfExtents(shape, aspect);
	const [px, py, pz] = worldPoint(position, shape, aspect);
	const planes: number[] = [];
	const quad = (corners: Vec3[], color: [number, number, number, number]) => {
		for (const i of [0, 1, 2, 0, 2, 3]) vertex(planes, corners[i]!, color);
	};
	quad(
		[
			[-hx, -hy, pz],
			[hx, -hy, pz],
			[hx, hy, pz],
			[-hx, hy, pz],
		],
		[0.25, 0.48, 0.95, 0.09],
	);
	quad(
		[
			[-hx, py, -hz],
			[hx, py, -hz],
			[hx, py, hz],
			[-hx, py, hz],
		],
		[0.25, 0.82, 0.5, 0.08],
	);
	quad(
		[
			[px, -hy, -hz],
			[px, hy, -hz],
			[px, hy, hz],
			[px, -hy, hz],
		],
		[0.95, 0.32, 0.35, 0.08],
	);

	const lines: number[] = [];
	const edge = (a: Vec3, b: Vec3, color: [number, number, number, number]) => {
		vertex(lines, a, color);
		vertex(lines, b, color);
	};
	const faint: [number, number, number, number] = [0.62, 0.68, 0.76, 0.5];
	for (const y of [-hy, hy]) for (const z of [-hz, hz]) edge([-hx, y, z], [hx, y, z], faint);
	for (const x of [-hx, hx]) for (const z of [-hz, hz]) edge([x, -hy, z], [x, hy, z], faint);
	for (const x of [-hx, hx]) for (const y of [-hy, hy]) edge([x, y, -hz], [x, y, hz], faint);
	edge([-hx, py, pz], [hx, py, pz], [0.95, 0.32, 0.35, 0.95]);
	edge([px, -hy, pz], [px, hy, pz], [0.25, 0.82, 0.5, 0.95]);
	edge([px, py, -hz], [px, py, hz], [0.25, 0.48, 0.95, 0.95]);
	return { planes: new Float32Array(planes), lines: new Float32Array(lines) };
}

/** CSS-pixel camera ray; its projection is the same as the rendered scene. */
export function minimapRay(position: Vec3, camera: MinimapCamera, shape: Vec3, aspect: Vec3, x: number, y: number, width: number, height: number): MeshRay {
	const { right, up, back } = cameraBasis(camera);
	const target = worldPoint(position, shape, aspect);
	const sx = (2 * x / width - 1) * width / height * Math.tan(HALF_FOV);
	const sy = (1 - 2 * y / height) * Math.tan(HALF_FOV);
	const direction = back.map((n, i) => -n + sx * right[i]! + sy * up[i]!) as Vec3;
	const length = Math.hypot(...direction);
	return { origin: target.map((n, i) => n + camera.distance * back[i]!) as Vec3, direction: direction.map((n) => n / length) as Vec3 };
}

export class MinimapRenderer {
	readonly gl: WebGL2RenderingContext;
	readonly program: WebGLProgram;
	readonly meshProgram: WebGLProgram;
	readonly buffer: WebGLBuffer;
	readonly surfaces: WebGLBuffer;
	readonly sceneArray: WebGLVertexArrayObject;
	readonly meshArray: WebGLVertexArrayObject;
	labels: Float32Array = new Float32Array();
	suggestions: Float32Array = new Float32Array();
	#combined = new Float32Array();
	#dirty = true;
	yaw = 0.72;
	pitch = 0.48;
	distance = 2.1;
	get suggestionCount(): number { return this.suggestions.length / MESH_STRIDE; }

	constructor(readonly canvas: HTMLCanvasElement, readonly shape: Vec3, readonly aspect: Vec3) {
		const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
		if (!gl) throw new Error("WebGL2 is needed for the 3D minimap");
		this.gl = gl;
		this.program = program(gl);
		this.meshProgram = program(gl, true);
		const buffer = gl.createBuffer(), surfaces = gl.createBuffer();
		const sceneArray = gl.createVertexArray(), meshArray = gl.createVertexArray();
		if (!buffer || !surfaces || !sceneArray || !meshArray) throw new Error("Couldn't allocate the 3D preview");
		this.buffer = buffer; this.surfaces = surfaces;
		this.sceneArray = sceneArray; this.meshArray = meshArray;
		gl.bindVertexArray(sceneArray);
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.enableVertexAttribArray(0); gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
		gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
		gl.bindVertexArray(meshArray);
		gl.bindBuffer(gl.ARRAY_BUFFER, surfaces);
		for (const [location, size, offset] of [[0, 3, 0], [1, 3, 12], [2, 3, 24], [3, 3, 36], [4, 4, 48], [5, 1, 64]] as const) {
			gl.enableVertexAttribArray(location);
			gl.vertexAttribPointer(location, size, gl.FLOAT, false, MESH_STRIDE * 4, offset);
			gl.vertexAttribDivisor(location, 1);
		}
		gl.bindVertexArray(null);
	}
	setMesh(layer: "labels" | "suggestions", data: Float32Array): void {
		this[layer] = data;
		this.#combined = new Float32Array(this.labels.length + this.suggestions.length);
		this.#combined.set(this.labels); this.#combined.set(this.suggestions, this.labels.length);
		this.#dirty = true;
	}
	pan(position: Vec3, dx: number, dy: number): Vec3 {
		return minimapPan(position, dx, dy, this.canvas.clientHeight, this, this.shape, this.aspect);
	}
	orbit(dx: number, dy: number): void {
		this.yaw -= dx * 0.008;
		this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + dy * 0.008));
		this.#dirty = true;
	}
	zoom(delta: number): void { this.distance = Math.max(1.15, Math.min(5, this.distance * Math.exp(delta * 0.001))); }

	pick(position: Vec3, x: number, y: number, opacity = { labels: 1, suggestions: 1 }): Vec3 | null {
		const ray = minimapRay(position, this, this.shape, this.aspect, x, y, this.canvas.clientWidth, this.canvas.clientHeight);
		let hit = opacity.labels > 0 ? intersectMesh(this.labels, ray) : null;
		if (opacity.suggestions > 0) hit = intersectMesh(this.suggestions, ray, hit?.distance) ?? hit;
		if (!hit) return null;
		const longest = Math.max(...this.shape.map((n, i) => n * this.aspect[i]!));
		return [hit.point[2] * longest / this.aspect[0] + this.shape[0] / 2,
			-hit.point[1] * longest / this.aspect[1] + this.shape[1] / 2,
			hit.point[0] * longest / this.aspect[2] + this.shape[2] / 2];
	}

	draw(position: Vec3, opacity = { labels: 1, suggestions: 1 }): void {
		const { gl } = this;
		gl.viewport(0, 0, this.canvas.width, this.canvas.height);
		gl.clearColor(0.045, 0.052, 0.064, 1);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
		gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
		gl.depthMask(false);
		const transform = minimapTransform(position, this, this.shape, this.aspect, this.canvas.width, this.canvas.height);
		const geometry = scene(this.shape, this.aspect, position);
		gl.useProgram(this.program);
		gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "transform"), false, transform);
		gl.bindVertexArray(this.sceneArray);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.bufferData(gl.ARRAY_BUFFER, geometry.planes, gl.DYNAMIC_DRAW);
		gl.drawArrays(gl.TRIANGLES, 0, geometry.planes.length / 7);
		if (this.#dirty) {
			gl.bindBuffer(gl.ARRAY_BUFFER, this.surfaces);
			gl.bufferData(gl.ARRAY_BUFFER, sortMesh(this.#combined, cameraBasis(this).back), gl.DYNAMIC_DRAW);
			this.#dirty = false;
		}
		if (this.#combined.length && (opacity.labels > 0 || opacity.suggestions > 0)) {
			gl.useProgram(this.meshProgram);
			gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProgram, "transform"), false, transform);
			gl.uniform2f(gl.getUniformLocation(this.meshProgram, "opacity"), opacity.labels, opacity.suggestions);
			gl.uniform1f(gl.getUniformLocation(this.meshProgram, "pixelRatio"), this.canvas.width / Math.max(1, this.canvas.clientWidth));
			gl.bindVertexArray(this.meshArray);
			gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, this.#combined.length / MESH_STRIDE);
		}
		gl.useProgram(this.program);
		gl.bindVertexArray(this.sceneArray);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.bufferData(gl.ARRAY_BUFFER, geometry.lines, gl.DYNAMIC_DRAW);
		gl.drawArrays(gl.LINES, 0, geometry.lines.length / 7);
		gl.depthMask(true);
		gl.bindVertexArray(null);
	}
	destroy(): void {
		this.gl.deleteBuffer(this.buffer); this.gl.deleteBuffer(this.surfaces);
		this.gl.deleteVertexArray(this.sceneArray); this.gl.deleteVertexArray(this.meshArray);
		this.gl.deleteProgram(this.program); this.gl.deleteProgram(this.meshProgram);
	}
}
