import * as zarr from "zarrita";
import { describe, expect, it } from "vitest";
import { useZstd } from "./zstd";

// np.arange(64, dtype="<u2"), compressed by numcodecs' Zstd (as the server
// writes chunks).
const FRAME =
	"KLUv/SCAbQIAAkgSCBD4bAYfq0wBe3l3dXNxb21raWdlY2FfXVtZV1VTUU9NS0lHRUNBPz07OTc1MzEvLSspJyUjIR8dGxkXFRMRDw0LCQcFAwG/AwA=";

describe("useZstd", () => {
	it("decodes the server's zstd chunks", async () => {
		useZstd();
		const Codec = await zarr.registry.get("zstd")?.();
		const codec = Codec?.fromConfig({ level: 0, checksum: false }, {} as never);
		const bytes = Uint8Array.from(atob(FRAME), (c) => c.charCodeAt(0));
		const decoded = (await codec?.decode(bytes)) as Uint8Array;
		expect(Array.from(new Uint16Array(decoded.buffer, decoded.byteOffset, 64))).toEqual(
			Array.from({ length: 64 }, (_, i) => i),
		);
	});
});
