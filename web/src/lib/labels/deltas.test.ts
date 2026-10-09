import { decompress } from "fzstd";
import { describe, expect, it } from "vitest";
import type { Vec3 } from "../viewer/tiles";
import fixtures from "../../../../tests/fixtures/labels/cases.json";
import { allowedValues, applyLocally, base64, fromBase64, packBits, splitIntoDeltas, unpackBits, zstdFrame } from "./deltas";

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
	if (name === "mixed") {
		// Every value 0 to 4 in blocks, as the fixtures' generator makes it.
		for (let z = 0; z < 64; z++) {
			for (let y = 0; y < 64; y++) {
				const value = (Math.floor(z / 8) + Math.floor(y / 8)) % 5;
				chunk.fill(value, (z * 64 + y) * 64, (z * 64 + y + 1) * 64);
			}
		}
	}
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

describe("allowedValues", () => {
	const listed = (allowed: Uint8Array) => [...allowed.keys()].filter((value) => allowed[value]);

	it("allows every value for any, and for a condition it doesn't know", () => {
		expect(listed(allowedValues("any"))).toHaveLength(256);
		expect(listed(allowedValues("something else"))).toHaveLength(256);
	});

	it("allows only unlabeled voxels for unlabeled, and every labeled one, background too, for labeled", () => {
		expect(listed(allowedValues("unlabeled"))).toEqual([0]);
		const labeled = listed(allowedValues("labeled"));
		expect(labeled).toHaveLength(255);
		expect(labeled).toContain(1);
		expect(labeled).not.toContain(0);
	});

	it("allows the values a class list names, in any order", () => {
		expect(listed(allowedValues("class:3"))).toEqual([3]);
		expect(listed(allowedValues("class:1,3"))).toEqual([1, 3]);
		expect(listed(allowedValues("class:5,2,3"))).toEqual([2, 3, 5]);
		expect(listed(allowedValues("class:007,254"))).toEqual([7, 254]);
	});

	it("allows every class value from 1 to 254 in a list that names them all", () => {
		const all = Array.from({ length: 254 }, (_, i) => i + 1);
		expect(listed(allowedValues(`class:${all.join(",")}`))).toEqual(all);
		expect(listed(allowedValues(`class:${[...all].reverse().join(",")}`))).toEqual(all);
	});

	it("puts no limit on a list it can't read, which the server refuses anyway", () => {
		for (const bad of ["class:", "class:2,", "class:,2", "class:a", "class: 2", "class:2,,3", "class:1000", "class:-1"]) {
			expect(listed(allowedValues(bad)), bad).toHaveLength(256);
		}
	});
});

describe("applyLocally with a condition", () => {
	/** A 1×1×12 row holding 0 (unlabeled), 1 (background), and classes 2 to 4, some twice. */
	const ROW = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 2, 0];

	/** The row after painting every voxel of it with `value` under `onlyIf`, and how many voxels changed. */
	function paint(onlyIf: string, value = 9, row = ROW) {
		const chunk = Uint8Array.from(row);
		const changed = applyLocally(chunk, [1, 1, row.length], [0, 0, 0, 1, 1, row.length], new Uint8Array(row.length).fill(1), value, onlyIf);
		return { row: [...chunk], changed };
	}

	it("paints anywhere", () => {
		expect(paint("any")).toEqual({ row: ROW.map(() => 9), changed: 12 });
	});

	it("paints only unlabeled voxels", () => {
		expect(paint("unlabeled").row).toEqual(ROW.map((v) => (v === 0 ? 9 : v)));
		expect(paint("unlabeled").changed).toBe(3);
	});

	it("paints only labeled voxels, background included", () => {
		expect(paint("labeled").row).toEqual(ROW.map((v) => (v === 0 ? 0 : 9)));
		expect(paint("labeled").changed).toBe(9);
	});

	it("paints only the one class", () => {
		expect(paint("class:2").row).toEqual(ROW.map((v) => (v === 2 ? 9 : v)));
		expect(paint("class:2").changed).toBe(3);
	});

	it("paints only the classes listed, background among them", () => {
		expect(paint("class:1,3").row).toEqual(ROW.map((v) => (v === 1 || v === 3 ? 9 : v)));
		expect(paint("class:3,1").changed).toBe(4);
		expect(paint("class:1,2,4").changed).toBe(7);
	});

	it("leaves everything alone when no voxel holds a listed value", () => {
		expect(paint("class:5,200")).toEqual({ row: ROW, changed: 0 });
	});

	it("erases only the classes listed", () => {
		expect(paint("class:2,4", 0).row).toEqual(ROW.map((v) => (v === 2 || v === 4 ? 0 : v)));
	});

	it("takes the values from a per-voxel list, under the same condition", () => {
		const chunk = Uint8Array.from(ROW);
		const values = Uint8Array.from(ROW.map((_, i) => 20 + i));
		const changed = applyLocally(chunk, [1, 1, 12], [0, 0, 0, 1, 1, 12], new Uint8Array(12).fill(1), values, "class:1,4");
		expect([...chunk]).toEqual(ROW.map((v, i) => (v === 1 || v === 4 ? 20 + i : v)));
		expect(changed).toBe(4);
	});
});
