import { expect, it, vi } from "vitest";
import { ChunkMeshCache, MeshDetailError, meshChunkInput } from "./chunk-mesh";
import { encodePreview, type SurfaceInput } from "./minimap-mesh";

function input(): SurfaceInput {
	return { level: { index: 0, path: "class", shape: [1, 1, 128], scale: [1, 1, 1] }, shape: [1, 1, 128], aspect: [1, 1, 1],
		colors: new Map([[2, "#ff0000"]]), chunks: [0, 1].map((x) => ({ id: `0/0/${x}`, chunk: { data: new Uint8Array(64).fill(2), shape: [1, 1, 64] } })) };
}
const triangle = () => Float32Array.from([2, 0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer;

it("sends only a core and positive halo slabs, not whole neighboring chunks", () => {
	const subset = meshChunkInput(input(), "0/0/0", 2);
	expect(subset.chunks.map((c) => c.chunk.shape)).toEqual([[1, 1, 64], [1, 1, 2]]);
	expect(encodePreview(subset).byteLength).toBe(16 + 48 + 66);
	expect(meshChunkInput(input(), "0/0/1", 1).chunks.map((c) => c.id)).toEqual(["0/0/1"]);
});

it("remeshes only edited cores and affected seam neighbors, and recolors from cached geometry", async () => {
	const cache = new ChunkMeshCache(), source = input(), fetch = vi.fn(async (_body: ArrayBuffer, _key: string, _factor: number) => triangle());
	await cache.build("labels", source, fetch);
	expect(fetch).toHaveBeenCalledTimes(2);
	await cache.build("labels", source, fetch);
	expect(fetch).toHaveBeenCalledTimes(2);
	(source.chunks[1]!.chunk.data as Uint8Array)[5] = 0;
	await cache.build("labels", source, fetch);
	expect(fetch).toHaveBeenCalledTimes(3);
	expect(fetch.mock.calls.at(-1)?.[1]).toBe("0,0,1");
	(source.chunks[1]!.chunk.data as Uint8Array)[0] = 0;
	await cache.build("labels", source, fetch);
	expect(fetch).toHaveBeenCalledTimes(5);
	const recolored = await cache.build("labels", { ...source, colors: new Map([[2, "#00ff00"]]), styles: { 2: { opacity: 0.5 } } }, fetch);
	expect(fetch).toHaveBeenCalledTimes(5);
	expect([...recolored.data.slice(12, 16)]).toEqual([0, 1, 0, expect.closeTo(0.39)]);
});

it("uses a common coarser detail after budget overflow, keeps it stable, and never mixes seam resolutions", async () => {
	const cache = new ChunkMeshCache(), source = input();
	const fetch = vi.fn(async (_body: ArrayBuffer, key: string, factor: number) => {
		if (key === "0,0,1" && factor < 4) throw new MeshDetailError();
		return triangle();
	});
	const result = await cache.build("suggestions", source, fetch);
	expect(result.downsample).toBe(4);
	expect(fetch.mock.calls.map(([, key, d]) => [key, d])).toEqual([["0,0,0", 1], ["0,0,1", 1], ["0,0,0", 2], ["0,0,1", 2], ["0,0,0", 4], ["0,0,1", 4]]);
	await cache.build("suggestions", source, fetch);
	expect(fetch).toHaveBeenCalledTimes(6);
});

it("does not treat network or permission errors as a reason to blur the mesh", async () => {
	const cache = new ChunkMeshCache(), fetch = vi.fn(async () => { throw new Error("Offline"); });
	await expect(cache.build("labels", input(), fetch)).rejects.toThrow("Offline");
	expect(fetch).toHaveBeenCalledOnce();
});
