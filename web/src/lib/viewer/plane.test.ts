import { describe, expect, it } from "vitest";
import { levelFactors } from "./image";
import { BACKGROUND_ALPHA } from "./background";
import { PlaneRenderer, paletteBytes, paletteRows, planeSlice, TextureCache } from "./plane";
import { PLANES, type TileKey } from "./tiles";

describe("planeSlice", () => {
	// 2 × 2 × 3 (z, y, x): value = 100 z + 10 y + x
	const values: number[] = [];
	for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 3; x++) values.push(100 * z + 10 * y + x);
	const chunk = { data: new Uint16Array(values), shape: [2, 2, 3] };

	it("cuts XY slices with x across and y down", () => {
		const slice = planeSlice(chunk, PLANES.xy, 1, Float32Array);
		expect([slice.width, slice.height]).toEqual([3, 2]);
		expect(Array.from(slice.data)).toEqual([100, 101, 102, 110, 111, 112]);
	});

	it("cuts XZ slices with x across and z down", () => {
		const slice = planeSlice(chunk, PLANES.xz, 1, Float32Array);
		expect([slice.width, slice.height]).toEqual([3, 2]);
		expect(Array.from(slice.data)).toEqual([10, 11, 12, 110, 111, 112]);
	});

	it("cuts YZ slices with z across and y down", () => {
		const slice = planeSlice(chunk, PLANES.yz, 2, Uint8Array);
		expect([slice.width, slice.height]).toEqual([2, 2]);
		expect(Array.from(slice.data)).toEqual([2, 102, 12, 112].map((v) => v & 255));
	});

	it("keeps 16-bit and negative values", () => {
		const data = new Int16Array([-30000, 30000]);
		expect(Array.from(planeSlice({ data, shape: [1, 1, 2] }, PLANES.xy, 0, Float32Array).data)).toEqual([
			-30000, 30000,
		]);
	});
});

describe("paletteBytes", () => {
	it("colors label values and leaves the rest clear", () => {
		const bytes = paletteBytes(new Map([[2, "#ff8000"], [3, "red"], [0, "#ffffff"]]));
		expect(Array.from(bytes.slice(8, 12))).toEqual([255, 128, 0, 255]);
		expect(Array.from(bytes.slice(0, 4))).toEqual([0, 0, 0, 0]);
		expect(Array.from(bytes.slice(12, 16))).toEqual([0, 0, 0, 0]);
	});
});

describe("paletteRows", () => {
	it("applies independent class opacity in both layers while Background stays labels-only", () => {
		const bytes = paletteRows(new Map([[1, "#ffffff"], [2, "#ff8000"], [3, "#ffffff"], [4, "#ffffff"], [255, "#ffffff"]]), {
			1: { color: "#123456", opacity: 0.5 }, 2: { opacity: 0.5 }, 3: { visible: false }, 4: { opacity: 0 },
		});
		for (const offset of [0, 1024]) {
			expect([...bytes.slice(offset + 8, offset + 12)]).toEqual([255, 128, 0, 128]);
			expect(bytes[offset + 3 * 4 + 3]).toBe(0);
			expect(bytes[offset + 4 * 4 + 3]).toBe(0);
			expect(bytes[offset + 255 * 4 + 3]).toBe(0);
		}
		expect([...bytes.slice(4, 8)]).toEqual([0, 0, 0, 0]);
		expect([...bytes.slice(1028, 1032)]).toEqual([0x12, 0x34, 0x56, 85]);
		expect(paletteRows(new Map(), { 1: { visible: false } })[1031]).toBe(0);
	});

	it("shows painted background as a haze in the labels' row only", () => {
		const bytes = paletteRows(new Map([[2, "#ff8000"]]));
		expect(bytes.length).toBe(256 * 4 * 2);
		// Classes look the same in both rows; background is clear in the first and a haze in the second.
		expect(Array.from(bytes.slice(8, 12))).toEqual([255, 128, 0, 255]);
		expect(Array.from(bytes.slice(1024 + 8, 1024 + 12))).toEqual([255, 128, 0, 255]);
		expect(Array.from(bytes.slice(4, 8))).toEqual([0, 0, 0, 0]);
		const [r, g, b, a] = bytes.slice(1024 + 4, 1024 + 8);
		expect(a).toBe(BACKGROUND_ALPHA);
		expect([r, g, b]).toEqual([0x7c, 0x8a, 0xa5]);
	});
});

describe("levelFactors", () => {
	it("gives each level's downsampling relative to level 0", () => {
		const scale = (s: number[]) => ({ path: "", coordinateTransformations: [{ type: "scale", scale: s }] });
		const factors = levelFactors({
			datasets: [scale([1, 4, 0.5, 0.5]), scale([1, 4, 1, 1]), scale([1, 8, 2, 2])],
		});
		expect(factors).toEqual([
			[1, 1, 1],
			[1, 2, 2],
			[2, 4, 4],
		]);
	});
});

/** A WebGL context that does nothing, counting the textures it's asked to delete, except as `overrides` say. */
function fakeGl(overrides: Record<string, unknown> = {}) {
	const deleted: unknown[] = [];
	let created = 0;
	const gl = new Proxy({} as Record<string, unknown>, {
		get(_target, name: string) {
			if (name in overrides) return overrides[name];
			if (name === "createTexture") return () => ({ texture: created++ });
			if (name === "deleteTexture") return (texture: unknown) => deleted.push(texture);
			if (name === "getShaderParameter" || name === "getProgramParameter") return () => true;
			// Constants are numbers; everything else a function that does nothing.
			return /^[A-Z0-9_]+$/.test(name) ? 1 : () => ({});
		},
	});
	return { gl, deleted };
}

describe("TextureCache", () => {
	const texture = (name: string) => ({ texture: name as unknown as WebGLTexture, width: 64, height: 64 });

	it("drops the least recently used textures past its limit, and deletes them on the GPU", () => {
		const { gl, deleted } = fakeGl();
		const cache = new TextureCache(gl as unknown as WebGL2RenderingContext, 2);
		cache.set("a", texture("a"));
		cache.set("b", texture("b"));
		cache.set("c", texture("c"));
		expect(cache.get("a")).toBeUndefined();
		expect(cache.get("b")).toBeDefined();
		expect(deleted).toEqual(["a"]);
	});

	it("counts a lookup as a use", () => {
		const { gl } = fakeGl();
		const cache = new TextureCache(gl as unknown as WebGL2RenderingContext, 2);
		cache.set("a", texture("a"));
		cache.set("b", texture("b"));
		cache.get("a");
		cache.set("c", texture("c"));
		expect(cache.get("a")).toBeDefined();
		expect(cache.get("b")).toBeUndefined();
	});

	it("replaces a texture under the same name, deleting the old one", () => {
		const { gl, deleted } = fakeGl();
		const cache = new TextureCache(gl as unknown as WebGL2RenderingContext, 4);
		cache.set("a", texture("old"));
		cache.set("a", texture("new"));
		expect(cache.get("a")?.texture).toBe("new");
		expect(deleted).toEqual(["old"]);
	});

	it("forgets textures by prefix, and all of them", () => {
		const { gl, deleted } = fakeGl();
		const cache = new TextureCache(gl as unknown as WebGL2RenderingContext, 8);
		for (const name of ["1/0@5", "1/1@5", "2/0@5"]) cache.set(name, texture(name));
		cache.deletePrefix("1/");
		expect(cache.get("1/0@5")).toBeUndefined();
		expect(cache.get("2/0@5")).toBeDefined();
		cache.clear();
		expect(cache.get("2/0@5")).toBeUndefined();
		expect(deleted).toHaveLength(3);
	});
});

describe("PlaneRenderer's textures", () => {
	const key = (cx: number): TileKey => ({ level: 0, cz: 0, cy: 0, cx });
	const chunk = { data: new Uint16Array(64 * 2 * 2), shape: [64, 2, 2] };
	const renderer = () => new PlaneRenderer({ getContext: () => fakeGl().gl } as unknown as HTMLCanvasElement, PLANES.xy, [64, 64, 64 * 4096]);

	/** One frame as the view does it: check each wanted chunk, upload the ones missing. */
	const frame = (r: PlaneRenderer, wanted: TileKey[]) => {
		for (const k of wanted) if (!r.hasImage(k, 0)) r.uploadImage(k, 0, chunk);
	};

	it("keeps the textures a frame checked for, whatever it uploads after", () => {
		const r = renderer();
		r.reserve(0, 0);
		// Earlier frames filled the cache (1024 textures), these the oldest.
		frame(r, Array.from({ length: 1024 }, (_, i) => key(i)));
		// This frame wants two of the oldest and two new ones, checking an old one first.
		const wanted = [key(0), key(2000), key(1), key(2001)];
		frame(r, wanted);
		for (const k of wanted) expect(r.hasImage(k, 0)).toBe(true);
	});

	it("keeps the label textures a frame checked for, whatever it uploads after", () => {
		const r = renderer();
		r.reserve(0, 0);
		const tile = (cx: number) => ({ id: `0/0/${cx}`, key: key(cx) });
		const labelFrame = (tiles: { id: string; key: TileKey }[]) => {
			for (const t of tiles) if (!r.hasLabels(t.id, 0)) r.uploadLabels(t, 0, chunk);
		};
		labelFrame(Array.from({ length: 1024 }, (_, i) => tile(i)));
		const wanted = [tile(0), tile(2000), tile(1), tile(2001)];
		labelFrame(wanted);
		for (const t of wanted) expect(r.hasLabels(t.id, 0)).toBe(true);
	});

	it("keeps room for what a frame asked for", () => {
		const r = renderer();
		const wanted = Array.from({ length: 1300 }, (_, i) => key(i));
		r.reserve(wanted.length, 0);
		frame(r, wanted);
		frame(r, wanted.slice(0, 5));
		for (const k of wanted) expect(r.hasImage(k, 0)).toBe(true);
	});
});

describe("PlaneRenderer's overlays", () => {
	/** A renderer on a context that has, or lacks, the stencil buffer it asked for, and the widths of the tiles it draws, in voxels. */
	const drawing = (stencil: boolean) => {
		const widths: number[] = [];
		let rect: number[] = [];
		const { gl } = fakeGl({
			getContextAttributes: () => ({ stencil }),
			getUniformLocation: (_program: unknown, name: string) => ({ name }),
			uniform4f: (location: { name: string }, ...values: number[]) => {
				if (location?.name === "rect") rect = values;
			},
			drawArrays: () => widths.push(rect[2]!),
		});
		const renderer = new PlaneRenderer({ getContext: () => gl } as unknown as HTMLCanvasElement, PLANES.xy, [128, 128, 128]);
		return { renderer, widths };
	};
	const chunk = { data: new Uint8Array(4), shape: [1, 2, 2] };
	// A tile of full resolution, and one of level 1 (voxels twice as big) over the same chunk.
	const fine = { id: "0/0/0", key: { level: 0, cz: 0, cy: 0, cx: 0 } };
	const coarse = { id: "1/0/0/0", key: { level: 1, cz: 0, cy: 0, cx: 0 }, scale: [2, 2, 2] as [number, number, number] };
	const view = { plane: PLANES.xy, position: [0, 0, 0] as [number, number, number], zoom: 1, aspect: [1, 1, 1] as [number, number, number], width: 100, height: 100 };

	const drawn = (stencil: boolean, level?: number) => {
		const { renderer, widths } = drawing(stencil);
		renderer.uploadLabels(fine, 0, chunk);
		renderer.uploadLabels(coarse, 0, chunk);
		renderer.draw(view, [0, 1], [], [{ slice: 0, tiles: [fine, coarse], opacity: 1, level }]);
		return widths;
	};

	it("draws the tiles of every level, finest first, where the canvas has a stencil buffer to keep them from drawing over each other", () => {
		expect(drawn(true, 1)).toEqual([2, 4]);
	});

	it("draws only the overlay's level where the canvas has no stencil buffer, for tiles of several levels would blend over each other", () => {
		expect(drawn(false, 1)).toEqual([4]);
		expect(drawn(false, 0)).toEqual([2]);
	});

	it("draws every tile of an overlay that names no level, stencil or not", () => {
		expect(drawn(false)).toEqual([2, 4]);
	});

	it("pulses stale chunks only, leaving fresh predictions and saved paint steady", () => {
		let alpha = 0;
		const drawnAlpha: number[] = [];
		const { gl } = fakeGl({
			getUniformLocation: (_program: unknown, name: string) => ({ name }),
			uniform1f: (location: { name: string }, value: number) => { if (location.name === "opacity") alpha = value; },
			drawArrays: () => drawnAlpha.push(alpha),
		});
		const renderer = new PlaneRenderer({ getContext: () => gl } as unknown as HTMLCanvasElement, PLANES.xy, [128, 128, 128]);
		const stale = { ...fine, id: "prediction/0/0/0", stale: true };
		const fresh = { ...fine, id: "prediction/0/0/1" };
		for (const tile of [stale, fresh, fine]) renderer.uploadLabels(tile, 0, chunk);
		for (const pulse of [0.5, 0.9]) renderer.draw(view, [0, 1], [], [
			{ slice: 0, tiles: [stale, fresh], opacity: 0.8, staleOpacity: pulse, hatched: true },
			{ slice: 0, tiles: [fine], opacity: 0.8 },
		]);
		expect(drawnAlpha).toEqual([0.4, 0.8, 0.8, expect.closeTo(0.72), 0.8, 0.8]);
	});
});
