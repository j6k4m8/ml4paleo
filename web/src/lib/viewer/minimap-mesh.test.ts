import { describe, expect, it } from "vitest";
import { decodePreview, encodePreview, intersectMesh, MESH_STRIDE, sortMesh, type SurfaceInput } from "./minimap-mesh";

const input: SurfaceInput = { level: { index: 0, path: "class", shape: [10, 20, 30], scale: [1, 1, 1] },
	chunks: [], colors: new Map([[2, "#ff0000"]]), shape: [10, 20, 30], aspect: [3, 1, 1] };
const triangle = (z = 5) => Float32Array.from([2, 0, 0, z, 30, 0, z, 0, 20, z]).buffer;

describe("zmesh preview wire format", () => {
	it("packs bounded chunks and masks hidden/background/declined values without editing input", () => {
		const data = Uint8Array.from([0, 1, 2, 3, 255, 2]);
		const source = { ...input, level: { ...input.level, shape: [1, 1, 6] as [number, number, number] },
			chunks: [{ id: "0/0/0", chunk: { data, shape: [1, 1, 6] } }], styles: { 3: { visible: false } } };
		const wire = encodePreview(source), header = new DataView(wire);
		expect([0, 4, 8, 12].map((i) => header.getUint32(i, true))).toEqual([1, 1, 6, 1]);
		expect([...new Uint8Array(wire, 40)]).toEqual([0, 0, 2, 0, 0, 2]);
		expect([...data]).toEqual([0, 1, 2, 3, 255, 2]);
	});
	it("clips padding and preserves nonzero chunk origins", () => {
		const wire = encodePreview({ ...input, level: { ...input.level, shape: [1, 1, 65] },
			chunks: [{ id: "0/0/1", chunk: { data: Uint8Array.of(2, 2), shape: [1, 1, 2] } }] });
		expect(wire.byteLength).toBe(41);
		expect(new DataView(wire).getUint32(24, true)).toBe(64);
	});
	it("converts native xyz vertices using physical proportions and styles", () => {
		const mesh = decodePreview(triangle(), { ...input, styles: { 2: { opacity: 0.5 } } }, true);
		expect(mesh).toHaveLength(MESH_STRIDE);
		expect([...mesh.slice(0, 9)]).toEqual([-0.5, expect.closeTo(1 / 3), 0, 0.5, expect.closeTo(1 / 3), 0, -0.5, expect.closeTo(-1 / 3), 0]);
		expect([...mesh.slice(12)]).toEqual([1, 0, 0, expect.closeTo(0.39), 1]);
	});
	it("maps partial pyramid cells without exceeding image bounds", () => {
		const source = { ...input, shape: [5, 7, 11] as [number, number, number], aspect: [6, 2, 1] as [number, number, number],
			level: { index: 1, path: "class_1", shape: [2, 4, 3] as [number, number, number], scale: [3, 2, 4] as [number, number, number] } };
		const mesh = decodePreview(Float32Array.from([2, 2, 3, 1, 3, 3, 1, 2.5, 4, 2]).buffer, source);
		expect(mesh[0]).toBeCloseTo(2.5 / 30);
		expect(mesh[3]).toBeCloseTo(5.5 / 30);
		expect(mesh[6]).toBeCloseTo(4 / 30); // x=2.5 is the center of partial [8,11], not [8,12].
		expect(mesh[8]).toBeCloseTo(0.5);
	});
	it("rejects malformed responses", () => {
		expect(() => decodePreview(new ArrayBuffer(7), input)).toThrow();
		expect(() => decodePreview(Float32Array.from([255, 0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer, input)).toThrow();
		expect(() => decodePreview(Float32Array.from([2, NaN, 0, 0, 1, 0, 0, 0, 1, 0]).buffer, input)).toThrow();
	});
});

describe("triangle raytracing and transparency", () => {
	it("hits the exact triangular surface, not its bounding rectangle", () => {
		const mesh = decodePreview(triangle(), input);
		const ray = { origin: [-0.25, 0.1, 2] as [number, number, number], direction: [0, 0, -1] as [number, number, number] };
		expect(intersectMesh(mesh, ray)).toEqual({ distance: 2, point: [-0.25, 0.1, 0] });
		expect(intersectMesh(mesh, { ...ray, origin: [0.4, -0.25, 2] })).toBeNull();
		expect(intersectMesh(mesh, { ...ray, direction: [1, 0, 0] })).toBeNull();
		expect(intersectMesh(mesh, ray, 1)).toBeNull();
		expect(intersectMesh(mesh, { origin: [-0.25, 0.1, -2], direction: [0, 0, 1] })?.distance).toBe(2);
	});
	it("selects the closest overlapping surface and ignores zero-opacity triangles", () => {
		const near = decodePreview(triangle(8), input, true), far = decodePreview(triangle(3), input);
		const combined = Float32Array.from([...far, ...near]);
		const ray = { origin: [-0.25, 0.1, 2] as [number, number, number], direction: [0, 0, -1] as [number, number, number] };
		expect(intersectMesh(combined, ray)?.point[2]).toBeCloseTo(0.3);
		combined[MESH_STRIDE + 15] = 0;
		expect(intersectMesh(combined, ray)?.point[2]).toBeCloseTo(-0.2);
	});
	it("sorts saved and hatched triangles together, without mutating inputs", () => {
		const combined = Float32Array.from([...decodePreview(triangle(8), input, true), ...decodePreview(triangle(3), input)]);
		const copy = combined.slice(), sorted = sortMesh(combined, [0, 0, 1]);
		expect(combined).toEqual(copy);
		expect(sorted[2]).toBeCloseTo(-0.2);
		expect(sorted[MESH_STRIDE + 2]).toBeCloseTo(0.3);
		expect(sorted[MESH_STRIDE + 16]).toBe(1);
	});
});
