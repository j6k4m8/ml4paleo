/**
 * A zstd codec for zarrita that decodes in plain JavaScript (fzstd). The
 * default one (numcodecs.js) builds functions at run time, which the app's
 * content security policy forbids.
 */

import { decompress } from "fzstd";
import * as zarr from "zarrita";

class ZstdCodec {
	kind = "bytes_to_bytes" as const;

	static fromConfig(): ZstdCodec {
		return new ZstdCodec();
	}

	encode(): never {
		throw new Error("The viewer only reads zstd chunks");
	}

	decode(bytes: Uint8Array): Uint8Array {
		return decompress(bytes);
	}
}

/** Decode zstd chunks with fzstd from now on. */
export function useZstd(): void {
	zarr.registry.set("zstd", async () => ZstdCodec);
}
