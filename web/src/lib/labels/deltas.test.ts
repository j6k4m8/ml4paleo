import { decompress } from "fzstd";
import { describe, expect, it } from "vitest";
import type { Vec3 } from "../viewer/tiles";
import fixtures from "../../../../tests/fixtures/labels/cases.json";
import { applyLocally, base64, fromBase64, packBits, splitIntoDeltas, unpackBits, zstdFrame } from "./deltas";

interface Cases {
	apply: {
		name: string;
		base: string;
		box: [number, number, number, number, number, number];
		mask_bits: string;
		value: number | null;
		values: string | null;
		only_if: string;
		expected: { class_sha256: string | null; changed: number };
	}[];
	split: {
		name: string;
		origin: Vec3;
		shape: Vec3;
		mask_bits: string;
		value: number;
		expected_deltas: { key: Vec3; box: number[]; mask_bits: string }[];
	}[];
}

const cases = fixtures as unknown as Cases;

async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
	const digest = await crypto.subtle.digest("SHA-256", bytes);
	return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

const volume = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

describe("zstdFrame", () => {
	it("makes frames a zstd decoder reads back", () => {
		const samples = [
			new Uint8Array(0),
			new Uint8Array([7]),
			new Uint8Array(300).fill(255),
			Uint8Array.from({ length: 70_000 }, (_, i) => (i % 1000 < 500 ? 0 : i & 255)),
			new Uint8Array(262_144).fill(3),
			Uint8Array.from({ length: 262_144 }, (_, i) => (i * 2654435761) >>> 24),
		];
		for (const data of samples) {
			const frame = zstdFrame(data);
			expect(decompress(frame)).toEqual(data);
			// The server refuses payloads more than 1024 bytes over their size.
			expect(frame.length).toBeLessThanOrEqual(data.length + 32);
		}
	});

	it("shrinks long runs", () => {
		expect(zstdFrame(new Uint8Array(32_768)).length).toBeLessThan(20);
	});
});

describe("splitIntoDeltas", () => {
	for (const c of cases.split) {
		it(`matches the fixture ${c.name}`, () => {
			const mask = unpackBits(fromBase64(c.mask_bits), volume(c.shape));
			const deltas = splitIntoDeltas(mask, c.shape, c.origin, { value: c.value });
			const got = deltas.map((d) => ({
				key: d.key,
				box: d.box,
				mask_bits: base64(packBits(unpackBits(decompress(fromBase64(d.mask)), volume(boxShape(d.box))))),
			}));
			const order = (d: { key: Vec3 }) => d.key.join("/");
			expect([...got].sort((a, b) => order(a).localeCompare(order(b)))).toEqual(
				[...c.expected_deltas].sort((a, b) => order(a).localeCompare(order(b))),
			);
			expect(deltas.every((d) => d.value === c.value && d.only_if === "any")).toBe(true);
		});
	}

	it("carries the base versions the client saw", () => {
		const mask = new Uint8Array([1, 1]);
		const [delta] = splitIntoDeltas(mask, [1, 1, 2], [0, 0, 63], { value: 2, baseVersions: new Map([["0/0/1", 7]]) }).slice(1);
		expect(delta?.key).toEqual([0, 0, 1]);
		expect(delta?.base_version).toBe(7);
	});
});

function boxShape(box: number[]): number[] {
	return [box[3]! - box[0]!, box[4]! - box[1]!, box[5]! - box[2]!];
}

function baseChunk(name: string): Uint8Array<ArrayBuffer> {
	const chunk = new Uint8Array(64 ** 3);
	if (name === "quadrants") {
		for (let z = 0; z < 64; z++) {
			for (let y = 0; y < 64; y++) {
				for (let x = 0; x < 64; x++) {
					if (z < 32 && y < 32) chunk[(z * 64 + y) * 64 + x] = 1;
					if (z >= 32 && y >= 32) chunk[(z * 64 + y) * 64 + x] = 3;
				}
			}
		}
	}
	return chunk;
}

describe("applyLocally", () => {
	for (const c of cases.apply) {
		it(`matches the server on ${c.name}`, async () => {
			const chunk = baseChunk(c.base);
			const mask = unpackBits(fromBase64(c.mask_bits), volume(boxShape(c.box)));
			const written = c.values ? fromBase64(c.values) : (c.value ?? 0);
			applyLocally(chunk, [64, 64, 64], c.box, mask, written, c.only_if);
			const sha = chunk.some((v) => v) ? await sha256(chunk) : null;
			expect(sha).toBe(c.expected.class_sha256);
		});
	}
});
