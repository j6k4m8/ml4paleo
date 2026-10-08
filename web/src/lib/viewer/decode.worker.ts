/**
 * Fetches and decodes chunks off the main thread. Each request names a zarr
 * array (a store URL and a path in it) and a region; the answer is the
 * region's values, transferred without copying (see decoding.ts).
 */

import { type CancelRequest, createDecoder, type DecodeRequest } from "./decoding";
import { useZstd } from "./zstd";

useZstd();

const decode = createDecoder();
const running = new Map<number, AbortController>();

self.onmessage = async (event: MessageEvent<DecodeRequest | CancelRequest>) => {
	const message = event.data;
	if (message.type === "cancel") {
		running.get(message.id)?.abort();
		return;
	}
	const controller = new AbortController();
	running.set(message.id, controller);
	try {
		const { reply, transfer } = await decode(message, controller.signal);
		(self as unknown as Worker).postMessage(reply, transfer);
	} finally {
		running.delete(message.id);
	}
};
