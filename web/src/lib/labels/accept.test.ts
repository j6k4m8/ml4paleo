import { decompress } from "fzstd";
import { describe, expect, it } from "vitest";
import type { Chunk } from "../viewer/chunks";
import { type Box, acceptParts, readBox } from "./accept";
import { fromBase64, unpackBits } from "./deltas";

/** A 100 × 70 × 130 volume whose value at (z, y, x) is (z + y + x) % 4. */
const SHAPE = [100, 70, 130];
function fakeChunk(id: string): Promise<Chunk> {
	const key = id.split("/").map(Number);
	const size = key.map((k, a) => Math.min(64, SHAPE[a]! - k * 64));
	const data = new Uint8Array(size[0]! * size[1]! * size[2]!);
	let i = 0;
	for (let z = 0; z < size[0]!; z++)
		for (let y = 0; y < size[1]!; y++)
			for (let x = 0; x < size[2]!; x++) data[i++] = (key[0]! * 64 + z + key[1]! * 64 + y + key[2]! * 64 + x) % 4;
	return Promise.resolve({ data, shape: size });
}

describe("accepting a prediction", () => {
	it("reads a box across chunks and edges", async () => {
		const box: Box = [60, 50, 120, 100, 70, 130];
		const values = await readBox(fakeChunk, box);
		expect(values.length).toBe(40 * 20 * 10);
		let i = 0;
		for (let z = 60; z < 100; z++)
			for (let y = 50; y < 70; y++) for (let x = 120; x < 130; x++) expect(values[i++]).toBe((z + y + x) % 4);
	});

	it("writes each value only into unlabeled voxels, never 0", async () => {
		const box: Box = [62, 0, 0, 66, 2, 2];
		const values = await readBox(fakeChunk, box);
		const parts = acceptParts(values, box);
		expect(parts).toHaveLength(3);
		const written = parts.flat();
		expect(written.every((d) => d.only_if === "unlabeled" && d.value !== 0)).toBe(true);
		expect(new Set(written.map((d) => d.value))).toEqual(new Set([1, 2, 3]));
		// Each part touches a chunk at most once.
		for (const part of parts) {
			const keys = part.map((d) => d.key.join("/"));
			expect(new Set(keys).size).toBe(keys.length);
		}
		const voxels = written.reduce((sum, d) => {
			const n = (d.box[3] - d.box[0]) * (d.box[4] - d.box[1]) * (d.box[5] - d.box[2]);
			return sum + unpackBits(decompress(fromBase64(d.mask)), n).reduce((a, b) => a + b, 0);
		}, 0);
		expect(voxels).toBe(values.filter((v) => v > 0).length);
	});
});
