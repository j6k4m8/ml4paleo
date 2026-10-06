/**
 * Fetches and decodes chunks off the main thread. Each request names a zarr
 * array (a store URL and a path in it) and a region; the answer is the
 * region's values, transferred without copying.
 */

import * as zarr from "zarrita";
import { useZstd } from "./zstd";

useZstd();

export type Region = [[number, number], [number, number], [number, number]];

export interface ArrayRegion {
	url: string;
	path: string;
	/** The (z, y, x) region to read. */
	region: Region;
	/** The channel to read, for arrays with a leading channel axis. */
	channel?: number;
}

export interface DecodeRequest extends ArrayRegion {
	type: "load";
	id: number;
}

export interface CancelRequest {
	type: "cancel";
	id: number;
}

export type DecodeResponse =
	| { id: number; data: ArrayBufferView; shape: number[] }
	| { id: number; error: string };

type OpenArray = zarr.Array<zarr.DataType, zarr.FetchStore>;

const arrays = new Map<string, Promise<OpenArray>>();
const running = new Map<number, AbortController>();

function openArray(url: string, path: string): Promise<OpenArray> {
	const key = `${url}#${path}`;
	let array = arrays.get(key);
	if (!array) {
		// Shard indexes are read with suffix ranges, which the gateway serves,
		// instead of a HEAD request first.
		const store = new zarr.FetchStore(url, { useSuffixRequest: true });
		array = zarr.open.v3(zarr.root(store).resolve(path), { kind: "array" });
		// Let a later request try again after a failure.
		array.catch(() => arrays.delete(key));
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
		const array = await openArray(message.url, message.path);
		const [[z0, z1], [y0, y1], [x0, x1]] = message.region;
		const spatial = [zarr.slice(z0, z1), zarr.slice(y0, y1), zarr.slice(x0, x1)];
		const selection = message.channel === undefined ? spatial : [message.channel, ...spatial];
		const chunk = await zarr.get(array, selection, { opts: { signal: controller.signal } });
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
