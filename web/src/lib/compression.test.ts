import { afterEach, describe, expect, it, vi } from "vitest";
import { compressedBody } from "./compression";

afterEach(() => vi.unstubAllGlobals());

describe("binary upload compression", () => {
	it("round-trips label bytes without changing the input", async () => {
		const raw = new Uint8Array(64 ** 3);
		raw.fill(2, 150, 12000);
		const copy = raw.slice();
		const wire = await compressedBody(raw.buffer);
		expect(wire.headers).toEqual({ "Content-Encoding": "gzip" });
		expect(wire.body.byteLength).toBeLessThan(raw.byteLength / 100);
		const decoded = await new Response(new Blob([wire.body]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
		expect(new Uint8Array(decoded)).toEqual(copy);
		expect(raw).toEqual(copy);
	});
	it("doesn't inflate tiny or incompressible data", async () => {
		for (const raw of [new Uint8Array(8), crypto.getRandomValues(new Uint8Array(32768))]) {
			const wire = await compressedBody(raw.buffer);
			expect(wire.body).toBe(raw.buffer);
			expect(wire.headers).toEqual({});
		}
	});
	it("works in browsers without native compression", async () => {
		vi.stubGlobal("CompressionStream", undefined);
		const raw = new ArrayBuffer(2048);
		expect(await compressedBody(raw)).toEqual({ body: raw, headers: {} });
	});
	it("falls back if a browser cannot create a gzip stream", async () => {
		vi.stubGlobal("CompressionStream", class { constructor() { throw new Error("unsupported"); } });
		const raw = new ArrayBuffer(2048);
		expect(await compressedBody(raw)).toEqual({ body: raw, headers: {} });
	});
});
