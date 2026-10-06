/**
 * Draws one plane of an image with WebGL2. Each 64×64 tile of the plane is a
 * float texture, windowed on the GPU so 16-bit and float images keep their
 * full range; labels draw over it as integer textures colored by a palette.
 */

import type { Chunk } from "./chunks";
import { CHUNK, type Level, type Plane, type TileKey, type Vec3, type View, pixelsPerVoxel, tileId } from "./tiles";

const VERTEX = `#version 300 es
in vec2 corner;
uniform vec4 rect;      // tile u, v, width, height in level-0 voxels
uniform vec2 uvMax;     // the part of the texture inside the image
uniform vec2 center;    // view center (u, v) in level-0 voxels
uniform vec2 toClip;    // clip-space units per level-0 voxel along u and v
out vec2 uv;
void main() {
	vec2 voxel = rect.xy + corner * rect.zw;
	vec2 clip = (voxel - center) * toClip;
	gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
	uv = corner * uvMax;
}`;

const IMAGE = `#version 300 es
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

const LABELS = `#version 300 es
precision highp float;
precision highp usampler2D;
in vec2 uv;
uniform usampler2D tile;
uniform sampler2D palette;  // 256 × 1, RGBA by label value
uniform float opacity;
out vec4 color;
void main() {
	ivec2 size = textureSize(tile, 0);
	ivec2 texel = min(ivec2(uv * vec2(size)), size - 1);
	uint value = texelFetch(tile, texel, 0).r;
	vec4 swatch = texelFetch(palette, ivec2(int(value), 0), 0);
	color = vec4(swatch.rgb, swatch.a * opacity);
}`;

interface Slice<T> {
	data: T;
	width: number;
	height: number;
}

/**
 * The 2D slice of a chunk at index `local` along the plane's normal, with
 * the plane's u axis across and v axis down.
 */
export function planeSlice<T extends Float32Array | Uint8Array>(
	chunk: Chunk,
	plane: Plane,
	local: number,
	make: new (length: number) => T,
): Slice<T> {
	const [dz = 1, dy = 1, dx = 1] = chunk.shape;
	const shape = [dz, dy, dx];
	const strides = [dy * dx, dx, 1];
	const width = shape[plane.u] ?? 1;
	const height = shape[plane.v] ?? 1;
	const out = new make(width * height);
	const source = chunk.data as unknown as ArrayLike<number | bigint>;
	const base = local * (strides[plane.normal] ?? 0);
	const su = strides[plane.u] ?? 0;
	const sv = strides[plane.v] ?? 0;
	for (let row = 0; row < height; row++) {
		for (let col = 0; col < width; col++) {
			out[row * width + col] = Number(source[base + row * sv + col * su]);
		}
	}
	return { data: out, width, height };
}

/** A 256-entry RGBA palette from label colors (#rrggbb) by value. */
export function paletteBytes(colors: Map<number, string>): Uint8Array {
	const bytes = new Uint8Array(256 * 4);
	for (const [value, color] of colors) {
		const match = /^#([0-9a-f]{6})$/i.exec(color);
		if (!match || value < 1 || value > 255) continue;
		const rgb = Number.parseInt(match[1] ?? "0", 16);
		bytes.set([rgb >> 16, (rgb >> 8) & 255, rgb & 255, 255], value * 4);
	}
	return bytes;
}

interface Texture {
	texture: WebGLTexture;
	width: number;
	height: number;
}

class TextureCache {
	#textures = new Map<string, Texture>();

	constructor(
		private gl: WebGL2RenderingContext,
		public limit: number,
	) {}

	get(id: string): Texture | undefined {
		const entry = this.#textures.get(id);
		if (entry) {
			this.#textures.delete(id);
			this.#textures.set(id, entry);
		}
		return entry;
	}

	has(id: string): boolean {
		return this.#textures.has(id);
	}

	set(id: string, entry: Texture): void {
		this.delete(id);
		this.#textures.set(id, entry);
		while (this.#textures.size > this.limit) {
			const oldest = this.#textures.keys().next().value as string;
			this.delete(oldest);
		}
	}

	delete(id: string): void {
		const entry = this.#textures.get(id);
		if (!entry) return;
		this.gl.deleteTexture(entry.texture);
		this.#textures.delete(id);
	}

	/** Forget every texture whose id starts with `prefix`. */
	deletePrefix(prefix: string): void {
		for (const id of [...this.#textures.keys()]) if (id.startsWith(prefix)) this.delete(id);
	}

	clear(): void {
		for (const id of [...this.#textures.keys()]) this.delete(id);
	}
}

/** Label-valued tiles drawn over the image with the palette. */
export interface Overlay {
	slice: number;
	tiles: LabelTile[];
	opacity: number;
}

export interface LabelTile {
	/** The label chunk id (`cz/cy/cx`) and its level-0 chunk key. */
	id: string;
	key: TileKey;
}

export class PlaneRenderer {
	#gl: WebGL2RenderingContext;
	#image: WebGLProgram;
	#labels: WebGLProgram;
	#imageUniforms: Record<string, WebGLUniformLocation | null> = {};
	#labelUniforms: Record<string, WebGLUniformLocation | null> = {};
	#imageTextures: TextureCache;
	#labelTextures: TextureCache;
	#palette: WebGLTexture;

	/**
	 * `extent` is the image's level-0 shape: coarse levels round their shape
	 * up, so their edge tiles are cut back to it.
	 */
	constructor(
		canvas: HTMLCanvasElement,
		private plane: Plane,
		private extent: Vec3,
	) {
		// No alpha channel: labels blend over the image, never with the page.
		const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
		if (!gl) throw new Error("This browser doesn't support WebGL2");
		this.#gl = gl;
		this.#image = this.#link(IMAGE);
		this.#labels = this.#link(LABELS);
		for (const name of ["rect", "uvMax", "center", "toClip", "tile", "window"]) {
			this.#imageUniforms[name] = gl.getUniformLocation(this.#image, name);
		}
		for (const name of ["rect", "uvMax", "center", "toClip", "tile", "palette", "opacity"]) {
			this.#labelUniforms[name] = gl.getUniformLocation(this.#labels, name);
		}
		const buffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
		for (const program of [this.#image, this.#labels]) {
			const corner = gl.getAttribLocation(program, "corner");
			gl.enableVertexAttribArray(corner);
			gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
		}
		gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
		this.#imageTextures = new TextureCache(gl, 1024);
		this.#labelTextures = new TextureCache(gl, 1024);
		this.#palette = this.#texture();
		this.setPalette(new Map());
	}

	#link(fragment: string): WebGLProgram {
		const gl = this.#gl;
		const program = gl.createProgram();
		for (const [type, source] of [
			[gl.VERTEX_SHADER, VERTEX],
			[gl.FRAGMENT_SHADER, fragment],
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
		gl.bindAttribLocation(program, 0, "corner");
		gl.linkProgram(program);
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			throw new Error(gl.getProgramInfoLog(program) ?? "Program didn't link");
		}
		return program;
	}

	#texture(): WebGLTexture {
		const gl = this.#gl;
		const texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		return texture;
	}

	/**
	 * Make room for at least `count` textures of each kind, so one frame's
	 * tiles never push each other out.
	 */
	reserve(images: number, labels: number): void {
		this.#imageTextures.limit = Math.max(1024, 2 * images);
		this.#labelTextures.limit = Math.max(1024, 2 * labels);
	}

	setPalette(colors: Map<number, string>): void {
		const gl = this.#gl;
		gl.bindTexture(gl.TEXTURE_2D, this.#palette);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, paletteBytes(colors));
	}

	/** Whether the image slice of `key` that the view cuts is on the GPU. */
	hasImage(key: TileKey, slice: number): boolean {
		return this.#imageTextures.has(`${tileId(key)}@${slice}`);
	}

	/** Upload the slice at level index `slice` of a loaded image chunk. */
	uploadImage(key: TileKey, slice: number, chunk: Chunk): void {
		const gl = this.#gl;
		const local = slice - [key.cz, key.cy, key.cx][this.plane.normal]! * CHUNK;
		const { data, width, height } = planeSlice(chunk, this.plane, local, Float32Array);
		const texture = this.#texture();
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
		this.#imageTextures.set(`${tileId(key)}@${slice}`, { texture, width, height });
	}

	hasLabels(id: string, slice: number): boolean {
		return this.#labelTextures.has(`${id}@${slice}`);
	}

	uploadLabels(tile: LabelTile, slice: number, chunk: Chunk): void {
		const gl = this.#gl;
		const local = slice - [tile.key.cz, tile.key.cy, tile.key.cx][this.plane.normal]! * CHUNK;
		const { data, width, height } = planeSlice(chunk, this.plane, local, Uint8Array);
		const texture = this.#texture();
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, data);
		this.#labelTextures.set(`${tile.id}@${slice}`, { texture, width, height });
	}

	/** Forget a label chunk's slices, for example after an edit. */
	dropLabels(id: string): void {
		this.#labelTextures.deletePrefix(`${id}@`);
	}

	/**
	 * Draw the view: for each image layer (coarsest first), the tiles on the
	 * GPU, so finer tiles cover coarser ones as they arrive; then each overlay
	 * (a model's prediction, the labels) in order.
	 */
	draw(
		view: View,
		window: [number, number],
		layers: { level: Level; slice: number; tiles: TileKey[] }[],
		overlays: Overlay[],
	): void {
		const gl = this.#gl;
		const { u, v } = this.plane;
		const px = pixelsPerVoxel(view);
		gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
		// The pasteboard around the image, as in the rest of the workspace.
		gl.clearColor(0x28 / 255, 0x28 / 255, 0x28 / 255, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
		const place = (uniforms: Record<string, WebGLUniformLocation | null>) => {
			gl.uniform2f(uniforms.center ?? null, view.position[u], view.position[v]);
			gl.uniform2f(uniforms.toClip ?? null, (2 * px[u]) / view.width, (2 * px[v]) / view.height);
		};
		const rect = (uniforms: Record<string, WebGLUniformLocation | null>, key: TileKey, scale: number[], entry: Texture) => {
			const c = [key.cz, key.cy, key.cx];
			const left = c[u]! * CHUNK * scale[u]!;
			const top = c[v]! * CHUNK * scale[v]!;
			const width = entry.width * scale[u]!;
			const height = entry.height * scale[v]!;
			const shownWidth = Math.max(0, Math.min(width, this.extent[u] - left));
			const shownHeight = Math.max(0, Math.min(height, this.extent[v] - top));
			gl.uniform4f(uniforms.rect ?? null, left, top, shownWidth, shownHeight);
			gl.uniform2f(uniforms.uvMax ?? null, shownWidth / width, shownHeight / height);
		};

		gl.disable(gl.BLEND);
		gl.useProgram(this.#image);
		place(this.#imageUniforms);
		gl.uniform2f(this.#imageUniforms.window ?? null, window[0], window[1]);
		gl.uniform1i(this.#imageUniforms.tile ?? null, 0);
		gl.activeTexture(gl.TEXTURE0);
		for (const { level, slice, tiles } of layers) {
			for (const key of tiles) {
				const entry = this.#imageTextures.get(`${tileId(key)}@${slice}`);
				if (!entry) continue;
				gl.bindTexture(gl.TEXTURE_2D, entry.texture);
				rect(this.#imageUniforms, key, level.scale, entry);
				gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
			}
		}

		gl.enable(gl.BLEND);
		gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
		gl.useProgram(this.#labels);
		place(this.#labelUniforms);
		gl.uniform1i(this.#labelUniforms.tile ?? null, 0);
		gl.uniform1i(this.#labelUniforms.palette ?? null, 1);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this.#palette);
		gl.activeTexture(gl.TEXTURE0);
		for (const overlay of overlays) {
			if (overlay.opacity <= 0) continue;
			gl.uniform1f(this.#labelUniforms.opacity ?? null, overlay.opacity);
			for (const tile of overlay.tiles) {
				const entry = this.#labelTextures.get(`${tile.id}@${overlay.slice}`);
				if (!entry) continue;
				gl.bindTexture(gl.TEXTURE_2D, entry.texture);
				rect(this.#labelUniforms, tile.key, [1, 1, 1], entry);
				gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
			}
		}
	}

	destroy(): void {
		this.#imageTextures.clear();
		this.#labelTextures.clear();
		this.#gl.deleteTexture(this.#palette);
		// Give the GPU context back now rather than at garbage collection.
		this.#gl.getExtension("WEBGL_lose_context")?.loseContext();
		this.#gl.deleteProgram(this.#image);
		this.#gl.deleteProgram(this.#labels);
	}
}
