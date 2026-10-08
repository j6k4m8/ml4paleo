import { decompress } from "fzstd";
import { describe, expect, it } from "vitest";
import type { Chunk } from "../viewer/chunks";
import { labelId, labelKey } from "../viewer/labels";
import { MAX_OVERLAY_TILES } from "../viewer/overlays";
import { PLANES, TILE_HYSTERESIS } from "../viewer/tiles";
import {
	type Box,
	MAX_ACCEPT_VOXELS,
	MAX_VIEW_CHUNKS,
	type ViewAccept,
	acceptParts,
	chunksIn,
	planeToAccept,
	readBox,
	tooBigForView,
	unlabeledOnly,
	whyNotInView,
} from "./accept";
import { CHUNK, type DeltaIn, decodeDelta, fromBase64, unpackBits } from "./deltas";

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

/** What a set of edits writes, by global voxel "z/y/x"; each voxel may be written once. */
function written(parts: DeltaIn[][]): Map<string, number> {
	const out = new Map<string, number>();
	for (const delta of parts.flat()) {
		const { mask, written: value } = decodeDelta(delta);
		const [z0, y0, x0, z1, y1, x1] = delta.box;
		let i = 0;
		for (let z = z0; z < z1; z++) {
			for (let y = y0; y < y1; y++) {
				for (let x = x0; x < x1; x++, i++) {
					if (!mask[i]) continue;
					const at = [delta.key[0] * CHUNK + z, delta.key[1] * CHUNK + y, delta.key[2] * CHUNK + x].join("/");
					expect(out.has(at), `${at} written twice`).toBe(false);
					out.set(at, value as number);
				}
			}
		}
	}
	return out;
}

describe("accepting what a view shows", () => {
	/** A slab's predicted values: 0 where nothing is predicted, and 1 (background), 2, and 3 in a pattern. */
	function slab(box: Box): Uint8Array {
		const size = [box[3] - box[0], box[4] - box[1], box[5] - box[2]];
		const values = new Uint8Array(size[0]! * size[1]! * size[2]!);
		for (let i = 0; i < values.length; i++) values[i] = (i * 7 + (i >> 3)) % 4;
		return values;
	}

	// One voxel thick along each axis, each across a chunk's edge along the axes it spans.
	const slabs: [string, Box][] = [
		["an XY slice", [70, 60, 60, 71, 70, 70]],
		["an XZ slice", [60, 9, 60, 70, 10, 70]],
		["a YZ slice", [60, 60, 127, 70, 70, 128]],
	];

	it.each(slabs)("writes what's predicted over %s, background too, never 0", (_name, box) => {
		const values = slab(box);
		const parts = acceptParts(values, box);
		const deltas = parts.flat();
		expect(deltas.length).toBeGreaterThan(0);
		expect(deltas.every((d) => d.only_if === "unlabeled" && d.value !== 0 && d.values === undefined)).toBe(true);
		expect(new Set(deltas.map((d) => d.value))).toEqual(new Set([1, 2, 3]));
		for (const part of parts) {
			// An op touches a chunk once, and carries one value.
			const keys = part.map((d) => d.key.join("/"));
			expect(new Set(keys).size).toBe(keys.length);
			expect(new Set(part.map((d) => d.value)).size).toBe(1);
		}
		// Exactly the predicted voxels, at their own places.
		const expected = new Map<string, number>();
		let i = 0;
		for (let z = box[0]; z < box[3]; z++) {
			for (let y = box[1]; y < box[4]; y++) {
				for (let x = box[2]; x < box[5]; x++, i++) if (values[i]) expected.set(`${z}/${y}/${x}`, values[i]!);
			}
		}
		expect(written(parts)).toEqual(expected);
		// Each delta stays inside its chunk.
		for (const d of deltas) {
			for (let a = 0; a < 3; a++) expect(d.box[a + 3]).toBeLessThanOrEqual(CHUNK);
		}
	});

	it("leaves out what's labeled already, whatever the label says", () => {
		const box: Box = [5, 0, 62, 6, 2, 66];
		const predicted = Uint8Array.from([1, 2, 3, 1, 2, 2, 2, 2]);
		// Hand labels over the first voxel (the prediction's value), the third (another value), and the last.
		const labeled = Uint8Array.from([1, 0, 4, 0, 0, 0, 0, 9]);
		const left = unlabeledOnly(predicted, labeled);
		expect([...left]).toEqual([0, 2, 0, 1, 2, 2, 2, 0]);
		// The input stays as it was, and what is left makes edits only for those voxels.
		expect([...predicted]).toEqual([1, 2, 3, 1, 2, 2, 2, 2]);
		const filled = written(acceptParts(left, box));
		expect(filled.size).toBe(5);
		expect(filled.has("5/0/62")).toBe(false);
		expect(filled.get("5/0/63")).toBe(2);
		expect(filled.get("5/1/65")).toBeUndefined();
		// Everything labeled: nothing to send.
		expect(acceptParts(unlabeledOnly(predicted, new Uint8Array(8).fill(1)), box)).toEqual([]);
		// Nothing labeled: all of it.
		expect([...unlabeledOnly(predicted, new Uint8Array(8))]).toEqual([...predicted]);
	});

	it("makes nothing from a slice where nothing is predicted", () => {
		const box: Box = [5, 0, 0, 6, 8, 8];
		expect(acceptParts(new Uint8Array(64), box)).toEqual([]);
	});

	it("asks only for full resolution chunks, never one of a coarser level of the labels", async () => {
		// The labels' store holds every level's chunks, a coarser level's with four-part ids.
		const asked: string[] = [];
		// Inside the 100 × 70 × 130 volume: two chunks across z, one along y, two along x.
		const box: Box = [62, 10, 60, 66, 60, 128];
		await readBox((id) => {
			asked.push(id);
			return fakeChunk(id);
		}, box);
		expect(asked.sort()).toEqual(["0/0/0", "0/0/1", "1/0/0", "1/0/1"]);
		expect(asked).toHaveLength(chunksIn(box));
		for (const id of asked) {
			expect(labelKey(id).level, id).toBe(0);
			expect(labelId(labelKey(id))).toBe(id);
		}
	});

	it("reads a slice from the chunks it crosses", async () => {
		const box: Box = [62, 0, 60, 63, 70, 70];
		const values = await readBox(fakeChunk, box);
		let i = 0;
		for (let y = 0; y < 70; y++) for (let x = 60; x < 70; x++) expect(values[i++]).toBe((62 + y + x) % 4);
	});

	it("means the only view there is, or in four views the one pointed at, else the one last used", () => {
		const { xy, xz, yz } = PLANES;
		// One view at a time: that one, whatever the pointer passed over.
		expect(planeToAccept("xz", yz, xy)).toBe(xz);
		expect(planeToAccept("yz", null, xy)).toBe(yz);
		expect(planeToAccept("xy", xz, yz)).toBe(xy);
		// Four views: the pointer's, and where it isn't over a view (on the panel, say), the last used,
		// not whichever it crossed on the way.
		expect(planeToAccept("four", yz, xy)).toBe(yz);
		expect(planeToAccept("four", null, xz)).toBe(xz);
	});

	it("counts the chunks a box touches", () => {
		expect(chunksIn([0, 0, 0, 1, 64, 64])).toBe(1);
		expect(chunksIn([0, 0, 0, 1, 65, 64])).toBe(2);
		expect(chunksIn([63, 63, 63, 65, 65, 65])).toBe(8);
		expect(chunksIn([64, 64, 64, 128, 128, 128])).toBe(1);
		expect(chunksIn([0, 0, 0, 256, 256, 256])).toBe(64);
		expect(chunksIn([1, 1, 1, 257, 257, 257])).toBe(125);
		expect(chunksIn([9, 0, 0, 10, 4096, 4096])).toBe(64 * 64);
	});

	it("is too big when it takes more chunks than a view surely draws the prediction from", () => {
		// A view draws it for up to 128 chunks, and again after being over only once it needs a fifth fewer.
		expect(MAX_OVERLAY_TILES).toBe(128);
		expect(MAX_VIEW_CHUNKS).toBe(Math.floor(MAX_OVERLAY_TILES / TILE_HYSTERESIS));
		expect(MAX_VIEW_CHUNKS).toBe(102);
		// 6 × 17 chunks of one slice.
		expect(tooBigForView([9, 0, 0, 10, 384, 1088])).toBe(false);
		expect(tooBigForView([9, 0, 0, 10, 384, 1089])).toBe(true);
		// A view just across a chunk's edge touches more than its area says.
		expect(tooBigForView([9, 1, 0, 10, 385, 1088])).toBe(true);
		expect(tooBigForView([9, 0, 0, 10, 4096, 4096])).toBe(true);
	});

	it("never reaches the server's limit with a view's one-voxel-thick slice, so only chunks are counted", () => {
		// The most voxels a slice of that many chunks holds is a chunk's 64 × 64 for each.
		expect(MAX_VIEW_CHUNKS * 64 * 64).toBeLessThan(MAX_ACCEPT_VOXELS);
	});

	describe("says why accepting in the view can't go ahead", () => {
		const slice: Box = [3, 0, 0, 4, 512, 512];
		const ready: ViewAccept = { imageReplaced: false, predicted: true, shown: true, opacity: 0.35, box: slice, mixed: false, covered: true };
		const why = (over: Partial<ViewAccept>) => whyNotInView({ ...ready, ...over });

		it("says nothing when it can", () => {
			expect(why({})).toBe("");
		});

		it("says what's wrong, most basic first", () => {
			expect(why({ imageReplaced: true })).toContain("image was replaced");
			expect(why({ predicted: false })).toContain("no prediction");
			expect(why({ shown: false })).toContain("hidden");
			expect(why({ opacity: 0 })).toContain("opacity is 0");
			expect(why({ opacity: Number.NaN })).toContain("opacity is 0");
			expect(why({ box: null })).toContain("Nothing of the image is in view");
			expect(why({ box: [3, 0, 0, 4, 4096, 4096] })).toBe("Zoom in a bit: the visible area is too big to accept at once.");
			expect(why({ mixed: true })).toContain("proposal");
			expect(why({ covered: false })).toContain("Nothing is predicted");
		});

		it("gives the first reason when there are several", () => {
			const everything = { imageReplaced: true, predicted: false, shown: false, opacity: 0, box: null, mixed: true, covered: false };
			expect(why(everything)).toContain("image was replaced");
			expect(why({ ...everything, imageReplaced: false })).toContain("no prediction");
			expect(why({ ...everything, imageReplaced: false, predicted: true })).toContain("hidden");
			expect(why({ ...everything, imageReplaced: false, predicted: true, shown: true })).toContain("opacity");
			expect(why({ ...everything, imageReplaced: false, predicted: true, shown: true, opacity: 1 })).toContain("Nothing of the image");
			// Too big comes before the proposal's edge, which comes before nothing being predicted.
			const big: Partial<ViewAccept> = { box: [3, 0, 0, 4, 4096, 4096], mixed: true, covered: false };
			expect(why(big)).toContain("too big");
			expect(why({ ...big, box: slice })).toContain("proposal");
			expect(why({ ...big, box: slice, mixed: false })).toContain("Nothing is predicted");
		});
	});
});
