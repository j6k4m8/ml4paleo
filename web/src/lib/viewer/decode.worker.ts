/**
 * Fetches and decodes image chunks off the main thread. Each request names
 * an OME-Zarr image (its URL), a level, and a chunk; the answer is the
 * chunk's voxels, transferred without copying.
 */

import * as zarr from "zarrita";
import { useZstd } from "./zstd";

useZstd();

export interface DecodeRequest {
	type: "load";
	id: number;
	url: string;
	level: number;
	region: [[number, number], [number, number], [number, number]];
}

export interface CancelRequest {
	type: "cancel";
	id: number;
}

export type DecodeResponse =
	| { id: number; data: ArrayBufferView; shape: number[] }
	| { id: number; error: string };

const arrays = new Map<string, Promise<zarr.Array<zarr.DataType, zarr.FetchStore>>>();
const running = new Map<number, AbortController>();

function openLevel(url: string, level: number) {
	const key = `${url}#${level}`;
	let array = arrays.get(key);
	if (!array) {
		// Shard indexes are read with suffix ranges, which the gateway serves,
		// instead of a HEAD request first.
		const store = new zarr.FetchStore(url, { useSuffixRequest: true });
		array = zarr.open.v3(zarr.root(store).resolve(String(level)), { kind: "array" });
		arrays.set(key, array);
	}
	return array;
}

self.onmessage = async (event: MessageEvent<DecodeRequest | CancelRequest>) => {
	const message = event.data;
	if (message.type === "cancel") {
		running.get(message.id)?.abort();
		return;
	}
	const controller = new AbortController();
	running.set(message.id, controller);
	try {
		const array = await openLevel(message.url, message.level);
		const [[z0, z1], [y0, y1], [x0, x1]] = message.region;
		const chunk = await zarr.get(
			array,
			[0, zarr.slice(z0, z1), zarr.slice(y0, y1), zarr.slice(x0, x1)],
			{ opts: { signal: controller.signal } },
		);
		const data = chunk.data as unknown as ArrayBufferView;
		const reply: DecodeResponse = { id: message.id, data, shape: chunk.shape };
		(self as unknown as Worker).postMessage(reply, [data.buffer as ArrayBuffer]);
	} catch (error) {
		const reply: DecodeResponse = { id: message.id, error: String(error) };
		(self as unknown as Worker).postMessage(reply);
	} finally {
		running.delete(message.id);
	}
};
