/** Chunk-local zmesh requests. Geometry is content-addressed, independent of color/alpha. */
import type { Chunk } from "./chunks";
import { labelKey } from "./labels";
import { decodePreview, encodePreview, MAX_MESH_TRIANGLES, type SurfaceInput } from "./minimap-mesh";
import type { MeshLayer } from "./minimap-meshing";

export class MeshDetailError extends Error {}
export type MeshFetch = (body: ArrayBuffer, key: string, downsample: number) => Promise<ArrayBuffer>;

/** Only the core and positive halo slabs affect this chunk's surface. An edit
 * inside another chunk cannot invalidate it; edits on shared faces can. */
export function meshChunkInput(input: SurfaceInput, id: string, halo: number): SurfaceInput {
	const own = labelKey(id), origin = [own.cz, own.cy, own.cx];
	return { ...input, chunks: input.chunks.flatMap((entry) => {
		const key = labelKey(entry.id), delta = [key.cz, key.cy, key.cx].map((n, a) => n - origin[a]!);
		if (delta.some((n) => n < 0 || n > 1)) return [];
		if (entry.id === id) return [entry];
		const shape = entry.chunk.shape.map((n, a) => delta[a] ? Math.min(halo, n) : n);
		const data = new Uint8Array(shape.reduce((a, b) => a * b, 1));
		let i = 0;
		const source = entry.chunk.data as Uint8Array;
		for (let z = 0; z < shape[0]!; z++) for (let y = 0; y < shape[1]!; y++) for (let x = 0; x < shape[2]!; x++) {
			data[i++] = source[(z * entry.chunk.shape[1]! + y) * entry.chunk.shape[2]! + x]!;
		}
		return [{ id: entry.id, chunk: { data, shape } satisfies Chunk }];
	}).sort((a, b) => a.id.localeCompare(b.id)) };
}

export class ChunkMeshCache {
	#cache = new Map<string, ArrayBuffer>();
	#detail = new Map<MeshLayer, { space: string; factor: number }>();
	async build(layer: MeshLayer, input: SurfaceInput, fetchMesh: MeshFetch): Promise<{ data: Float32Array; downsample: number }> {
		if (input.chunks.length > 16) throw new Error("Too many 3D preview chunks.");
		const space = `${input.level.index}:${input.level.shape.join()}`;
		const previous = this.#detail.get(layer);
		let factor = previous?.space === space ? previous.factor : 1;
		// One resolution per layer keeps adjoining chunk seams compatible.
		// Detail can fall under load, but doesn't oscillate on every brush stroke.
		for (; factor <= 16; factor *= 2) {
			try {
				const pieces: Float32Array[] = [];
				for (const { id } of input.chunks) {
					const raw = encodePreview(meshChunkInput(input, id, factor));
					const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", raw))].map((n) => n.toString(16).padStart(2, "0")).join("");
					const key = `${id}:${factor}:${hash}`;
					let wire = this.#cache.get(key);
					if (wire) this.#cache.delete(key);
					else {
						const at = labelKey(id);
						wire = await fetchMesh(raw, `${at.cz},${at.cy},${at.cx}`, factor);
						if (wire.byteLength / 40 > MAX_MESH_TRIANGLES / 16) throw new MeshDetailError();
					}
					const decoded = decodePreview(wire, input, layer === "suggestions");
					// At most 64 × 300 KB of wire geometry across both layers/LODs.
					this.#cache.set(key, wire);
					while (this.#cache.size > 64) this.#cache.delete(this.#cache.keys().next().value!);
					pieces.push(decoded);
				}
				this.#detail.set(layer, { space, factor });
				const data = new Float32Array(pieces.reduce((n, p) => n + p.length, 0));
				let offset = 0;
				for (const piece of pieces) { data.set(piece, offset); offset += piece.length; }
				return { data, downsample: factor };
			} catch (error) {
				if (!(error instanceof MeshDetailError)) throw error;
			}
		}
		throw new Error("Couldn't simplify the 3D preview. Previous surfaces are still shown.");
	}
}
