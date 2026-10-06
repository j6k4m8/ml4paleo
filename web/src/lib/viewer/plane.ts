/**
 * Draws one z plane of an image with WebGL2: each 64×64 tile of the plane is
 * a float texture, windowed on the GPU so 16-bit and float images keep their
 * full range.
 */

import type { Chunk } from "./chunks";
import { CHUNK, type Level, type TileKey, type View, tileId } from "./tiles";

const VERTEX = `#version 300 es
in vec2 corner;
uniform vec4 rect;      // tile x, y, width, height in level-0 voxels
uniform vec2 center;    // view center (x, y) in level-0 voxels
uniform vec2 halfSize;  // half the canvas, in level-0 voxels
out vec2 uv;
void main() {
	vec2 voxel = rect.xy + corner * rect.zw;
	vec2 ndc = (voxel - center) / halfSize;
	gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
	uv = corner;
}`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 uv;
uniform sampler2D tile;
uniform vec2 window;    // low, high
out vec4 color;
void main() {
	float value = texture(tile, uv).r;
	float shade = clamp((value - window.x) / max(window.y - window.x, 1e-20), 0.0, 1.0);
	color = vec4(vec3(shade), 1.0);
}`;

/** The 2D slice at local z `lz` of a chunk, as floats (y-major). */
export function planeSlice(chunk: Chunk, lz: number): { data: Float32Array; width: number; height: number } {
	const [, height = 0, width = 0] = chunk.shape;
	const start = lz * height * width;
	const out = new Float32Array(height * width);
	const source = chunk.data as unknown as ArrayLike<number | bigint>;
	for (let i = 0; i < out.length; i++) out[i] = Number(source[start + i]);
	return { data: out, width, height };
}

interface Texture {
	texture: WebGLTexture;
	width: number;
	height: number;
}

export class PlaneRenderer {
	#gl: WebGL2RenderingContext;
	#program: WebGLProgram;
	#textures = new Map<string, Texture>();
	#uniforms: Record<string, WebGLUniformLocation | null> = {};
	maxTextures = 1024;

	constructor(canvas: HTMLCanvasElement) {
		const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
		if (!gl) throw new Error("This browser doesn't support WebGL2");
		this.#gl = gl;
		this.#program = this.#link();
		for (const name of ["rect", "center", "halfSize", "tile", "window"]) {
			this.#uniforms[name] = gl.getUniformLocation(this.#program, name);
		}
		const buffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
		const corner = gl.getAttribLocation(this.#program, "corner");
		gl.enableVertexAttribArray(corner);
		gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
	}

	#link(): WebGLProgram {
		const gl = this.#gl;
		const program = gl.createProgram();
		for (const [type, source] of [
			[gl.VERTEX_SHADER, VERTEX],
			[gl.FRAGMENT_SHADER, FRAGMENT],
		] as const) {
			const shader = gl.createShader(type);
			if (!shader) throw new Error("Couldn't create a shader");
			gl.shaderSource(shader, source);
			gl.compileShader(shader);
			if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
				throw new Error(gl.getShaderInfoLog(shader) ?? "Shader didn't compile");
			}
			gl.attachShader(program, shader);
		}
		gl.linkProgram(program);
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			throw new Error(gl.getProgramInfoLog(program) ?? "Program didn't link");
		}
		return program;
	}

	/** Whether the slice of `key` at level-0 plane `z` is on the GPU. */
	has(key: TileKey, z: number, level: Level): boolean {
		return this.#textures.has(textureId(key, z, level));
	}

	/** Upload the slice of a loaded chunk that plane `z` cuts through. */
	upload(key: TileKey, z: number, level: Level, chunk: Chunk): void {
		const id = textureId(key, z, level);
		if (this.#textures.has(id)) return;
		const gl = this.#gl;
		const lz = Math.floor(z / level.scale[0]) - key.cz * CHUNK;
		const slice = planeSlice(chunk, lz);
		const texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, slice.width, slice.height, 0, gl.RED, gl.FLOAT, slice.data);
		this.#textures.set(id, { texture, width: slice.width, height: slice.height });
		while (this.#textures.size > this.maxTextures) {
			const [oldest, entry] = this.#textures.entries().next().value as [string, Texture];
			gl.deleteTexture(entry.texture);
			this.#textures.delete(oldest);
		}
	}

	/**
	 * Draw the view: for each level in `layers` (coarsest first), the tiles
	 * that are on the GPU, so finer tiles cover coarser ones as they arrive.
	 */
	draw(view: View, window: [number, number], layers: { level: Level; tiles: TileKey[] }[]): void {
		const gl = this.#gl;
		gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
		gl.clearColor(0, 0, 0, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
		gl.useProgram(this.#program);
		gl.uniform2f(this.#uniforms.center ?? null, view.centerX, view.centerY);
		gl.uniform2f(this.#uniforms.halfSize ?? null, view.width / 2 / view.zoom, view.height / 2 / view.zoom);
		gl.uniform2f(this.#uniforms.window ?? null, window[0], window[1]);
		gl.uniform1i(this.#uniforms.tile ?? null, 0);
		gl.activeTexture(gl.TEXTURE0);
		for (const { level, tiles } of layers) {
			const [, sy, sx] = level.scale;
			for (const key of tiles) {
				const id = textureId(key, view.z, level);
				const entry = this.#textures.get(id);
				if (!entry) continue;
				// Mark as just used.
				this.#textures.delete(id);
				this.#textures.set(id, entry);
				gl.bindTexture(gl.TEXTURE_2D, entry.texture);
				gl.uniform4f(
					this.#uniforms.rect ?? null,
					key.cx * CHUNK * sx,
					key.cy * CHUNK * sy,
					entry.width * sx,
					entry.height * sy,
				);
				gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
			}
		}
	}

	destroy(): void {
		for (const { texture } of this.#textures.values()) this.#gl.deleteTexture(texture);
		this.#textures.clear();
		this.#gl.deleteProgram(this.#program);
	}
}

function textureId(key: TileKey, z: number, level: Level): string {
	return `${tileId(key)}@${Math.floor(z / level.scale[0])}`;
}
