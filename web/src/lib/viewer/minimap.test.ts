import { describe, expect, it, vi } from "vitest";
import { minimapIds, minimapLevel, minimapPan, minimapRay, minimapSuggestionIds, suggestionChunks, minimapTransform, MinimapRenderer, worldHalfExtents, worldPoint } from "./minimap";
import { decodePreview, MESH_STRIDE } from "./minimap-mesh";
import type { Box } from "../rois.svelte";
import type { Chunk } from "./chunks";
import { ViewerState } from "./state.svelte";
import type { Level, Vec3 } from "./tiles";

describe("bounded minimap inputs", () => {
	it("chooses the finest bounded coarse level", () => {
		const levels: Level[] = [
			{ index: 0, path: "class", shape: [512, 512, 1024], scale: [1, 1, 1] },
			{ index: 1, path: "class_1", shape: [256, 256, 512], scale: [2, 2, 2] },
			{ index: 2, path: "class_2", shape: [128, 128, 256], scale: [4, 4, 4] },
		];
		const level = minimapLevel(levels)!;
		expect(level.index).toBe(2);
		const ids = minimapIds(level);
		expect(ids).toHaveLength(16);
		expect(ids[0]).toBe("2/0/0/0");
		expect(ids.at(-1)).toBe("2/1/1/3");
	});
	it("preserves anisotropy and centers the volume", () => {
		expect(worldHalfExtents([10, 20, 30], [3, 1, 1])).toEqual([0.5, 1 / 3, 0.5]);
		expect(worldPoint([5, 10, 15], [10, 20, 30], [3, 1, 1])).toEqual([0, 0, 0]);
	});
});

describe("nearby suggestion mesh inputs", () => {
	const colors = new Map([[2, "#ff0000"], [3, "#00ff00"]]);
	const chunk = (values: number[]): Chunk => ({ data: Uint8Array.from(values), shape: [1, 1, values.length] });
	const shape: Vec3 = [1, 1, 8];
	const box: Box = [0, 0, 0, ...shape];
	const map = (values: number[]) => new Map([["0/0/0", chunk(values)]]);
	it("bounds whole-volume reads around the center", () => {
		const huge: Vec3 = [1e9, 1e9, 1e9];
		const ids = minimapSuggestionIds(huge, [5000, 5000, 5000], [{ box: [0, 0, 0, ...huge] }]);
		expect(ids).toHaveLength(16);
		expect(ids[0]).toBe("78/78/78");
	});
	it("reads only ready Live chunks and respects proposal bounds", () => {
		expect(minimapSuggestionIds([128, 128, 128], [64, 64, 64], [{ box: [0, 0, 0, 128, 128, 128], keys: ["0/0/0", "9/9/9"] }])).toEqual(["0/0/0"]);
		expect(minimapSuggestionIds([1000, 1000, 1000], [0, 0, 0], [{ box: [800, 800, 800, 801, 801, 801] }])).toEqual(["12/12/12"]);
		expect(minimapSuggestionIds(shape, [0, 0, 0], [{ box, keys: [] }])).toEqual([]);
	});
	it("masks all saved state, even background, declines, and hidden labels", () => {
		const chunks = suggestionChunks(["0/0/0"], [{ box, chunks: map([2, 2, 2, 2, 3, 1, 255, 7]) }],
			map([0, 1, 2, 255, 0, 0, 0, 0]), colors);
		expect([...chunks[0]!.chunk.data as Uint8Array]).toEqual([2, 0, 0, 0, 3, 0, 0, 0]);
	});
	it("fails closed until the saved mask loads", () => {
		expect(suggestionChunks(["0/0/0"], [{ box, chunks: map(new Array(8).fill(2)) }], new Map(), colors)).toEqual([]);
	});
	it("lets the newest proposal own its box, even Background and not-yet-loaded regions", () => {
		const proposal = { box: [0, 0, 2, 1, 1, 5] as Box, chunks: map([3, 3, 3, 1, 0, 3, 3, 3]) };
		const whole = { box, chunks: map(new Array(8).fill(2)) };
		const mask = map(new Array(8).fill(0));
		expect([...suggestionChunks(["0/0/0"], [proposal, whole], mask, colors)[0]!.chunk.data as Uint8Array]).toEqual([2, 2, 3, 0, 0, 2, 2, 2]);
		expect([...suggestionChunks(["0/0/0"], [{ ...proposal, chunks: new Map() }, whole], mask, colors)[0]!.chunk.data as Uint8Array]).toEqual([2, 2, 0, 0, 0, 2, 2, 2]);
	});
});

function project(point: Vec3, transform: Float32Array): Vec3 {
	const v = [...point, 1];
	const clip = [0, 1, 2, 3].map((row) => v.reduce((n, value, column) => n + value * transform[column * 4 + row]!, 0));
	return clip.slice(0, 3).map((n) => n / clip[3]!) as Vec3;
}

describe("synchronized minimap navigation", () => {
	const shape: Vec3 = [40, 80, 120], aspect: Vec3 = [5, 2, 1];
	const position: Vec3 = [13, 27, 77];
	const cameras = [
		{ yaw: 0, pitch: 0, distance: 2.1 },
		{ yaw: 0.72, pitch: 0.48, distance: 1.15 },
		{ yaw: -2.1, pitch: -1.2, distance: 5 },
	];
	it.each(cameras)("centers every shared XYZ position under camera $yaw / $pitch", (camera) => {
		for (const point of [position, [0, 0, 0], [39.5, 79.5, 119.5]] as Vec3[]) {
			const projected = project(worldPoint(point, shape, aspect), minimapTransform(point, camera, shape, aspect, 600, 300));
			expect(projected[0]).toBeCloseTo(0);
			expect(projected[1]).toBeCloseTo(0);
		}
	});
	it.each(cameras)("makes a grabbed point follow screen drag under camera $yaw / $pitch", (camera) => {
		const dx = 21, dy = -17, width = 600, height = 300;
		const next = minimapPan(position, dx, dy, height, camera, shape, aspect);
		const projected = project(worldPoint(position, shape, aspect), minimapTransform(next, camera, shape, aspect, width, height));
		expect(projected[0] * width / 2).toBeCloseTo(dx, 4);
		expect(-projected[1] * height / 2).toBeCloseTo(dy, 4);
		// Returning the same CSS distance is reversible despite anisotropy.
		expect(minimapPan(next, -dx, -dy, height, camera, shape, aspect)).toEqual(position.map((n) => expect.closeTo(n)));
	});
	it("uses shared ViewerState bounds when a 3D pan reaches the image edge", () => {
		const viewer = new ViewerState(shape, aspect);
		viewer.moveTo(minimapPan(position, 10000, -10000, 300, cameras[0]!, shape, aspect));
		expect(viewer.position).toEqual([13, 79.5, 0]);
	});
});


describe("surface camera rays", () => {
	it.each([{ yaw: 0, pitch: 0, distance: 2 }, { yaw: 1.2, pitch: -0.8, distance: 3 }])("inverts the renderer projection for $yaw", (camera) => {
		const shape: Vec3 = [30, 40, 50], aspect: Vec3 = [7, 2, 1], position: Vec3 = [4, 11, 35];
		const transform = minimapTransform(position, camera, shape, aspect, 900, 300);
		for (const [x, y] of [[450, 150], [130, 95], [620, 230]]) {
			const ray = minimapRay(position, camera, shape, aspect, x!, y!, 900, 300);
			const p = ray.origin.map((n, i) => n + 2 * ray.direction[i]!) as Vec3;
			const projected = project(p, transform);
			expect((projected[0] + 1) * 450).toBeCloseTo(x!, 3);
			expect((1 - projected[1]) * 150).toBeCloseTo(y!, 3);
		}
	});
});

describe("translucent surface renderer", () => {
	function setup() {
		const calls = {
			bufferData: vi.fn(), vertexAttribPointer: vi.fn(), vertexAttribDivisor: vi.fn(),
			drawArraysInstanced: vi.fn(), uniformMatrix4fv: vi.fn(), uniform2f: vi.fn(), depthMask: vi.fn(),
			deleteBuffer: vi.fn(), deleteVertexArray: vi.fn(), deleteProgram: vi.fn(),
		};
		const gl = new Proxy(calls as Record<string, unknown>, {
			get(target, key: string) {
				if (key in target) return target[key];
				if (key === "getShaderParameter" || key === "getProgramParameter") return () => true;
				return /^[A-Z0-9_]+$/.test(key) ? key : () => ({});
			},
		}) as unknown as WebGL2RenderingContext;
		const canvas = { getContext: () => gl, width: 1000, height: 600, clientWidth: 500, clientHeight: 300 } as unknown as HTMLCanvasElement;
		return { renderer: new MinimapRenderer(canvas, [10, 20, 30], [3, 1, 1]), calls, gl };
	}
	it("draws merged faces in one shared translucent batch with the scene transform", () => {
		const { renderer, calls, gl } = setup();
		renderer.setMesh("labels", new Float32Array(2 * MESH_STRIDE));
		renderer.setMesh("suggestions", new Float32Array(3 * MESH_STRIDE));
		renderer.draw([2, 4, 7], { labels: 0.5, suggestions: 0.3 });
		expect(calls.vertexAttribPointer).toHaveBeenCalledWith(2, 3, gl.FLOAT, false, 68, 24);
		for (const location of [0, 1, 2, 3, 4, 5]) expect(calls.vertexAttribDivisor).toHaveBeenCalledWith(location, 1);
		expect(calls.drawArraysInstanced).toHaveBeenCalledWith(gl.TRIANGLES, 0, 3, 5);
		expect(calls.uniform2f.mock.calls[0]!.slice(1)).toEqual([0.5, 0.3]);
		expect(calls.uniformMatrix4fv.mock.calls[0]![2]).toEqual(calls.uniformMatrix4fv.mock.calls[1]![2]);
		expect(calls.depthMask.mock.calls.map((c) => c[0])).toEqual([false, true]);
		renderer.destroy();
		expect(calls.deleteBuffer).toHaveBeenCalledTimes(2);
		expect(calls.deleteVertexArray).toHaveBeenCalledTimes(2);
		expect(calls.deleteProgram).toHaveBeenCalledTimes(2);
	});
	it("does not sort or upload surfaces again during pan, zoom, or opacity changes", () => {
		const { renderer, calls } = setup();
		renderer.setMesh("labels", new Float32Array(2 * MESH_STRIDE));
		renderer.draw([5, 10, 15]);
		calls.bufferData.mockClear();
		renderer.zoom(50);
		renderer.draw([4, 9, 14], { labels: 0, suggestions: 1 });
		expect(calls.bufferData).toHaveBeenCalledTimes(2); // Context planes/lines only.
		renderer.orbit(10, 20); renderer.draw([4, 9, 14]);
		expect(calls.bufferData).toHaveBeenCalledTimes(5);
	});
	it("picks the nearest visible surface in shared zyx units and respects layer hiding", () => {
		const { renderer } = setup();
		const shape = renderer.shape;
		const mesh = decodePreview(Float32Array.from([2, 0, 0, 8, 30, 0, 8, 0, 20, 8]).buffer,
			{ level: { index: 0, path: "class", shape, scale: [1, 1, 1] }, chunks: [], colors: new Map(), shape, aspect: renderer.aspect });
		renderer.setMesh("suggestions", mesh);
		renderer.yaw = 0; renderer.pitch = 0;
		expect(renderer.pick([5, 10, 15], 250, 150)).toEqual([expect.closeTo(8), expect.closeTo(10), expect.closeTo(15)]);
		expect(renderer.pick([5, 10, 15], 250, 150, { labels: 1, suggestions: 0 })).toBeNull();
		expect(renderer.pick([5, 10, 15], 0, 0)).toBeNull();
	});
	it("preserves orbit direction and CSS-pixel pan", () => {
		const { renderer } = setup();
		renderer.orbit(10, 20);
		expect(renderer.yaw).toBeCloseTo(0.64);
		expect(renderer.pitch).toBeCloseTo(0.64);
		expect(renderer.pan([5, 10, 15], 10, 20)).toEqual(minimapPan([5, 10, 15], 10, 20, 300, renderer, renderer.shape, renderer.aspect));
	});
});
