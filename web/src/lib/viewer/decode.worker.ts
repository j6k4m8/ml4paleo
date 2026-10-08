/**
 * Fetches and decodes chunks off the main thread. Each request names a zarr
 * array (a store URL and a path in it) and a region; the answer is the
 * region's values, transferred without copying.
 */

import * as zarr from "zarrita";
import { failure, shardedFetch } from "./image";
import { patientFetch } from "./patient";
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
	/**
	 * An array whose chunks the server makes when asked for (a level of the
	 * label pyramid): a 503 is asked again, and a chunk's version is the
	 * `X-Pyramid-Version` it was made from, not an edit's `X-Chunk-Version`.
	 */
	derived?: boolean;
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
	| { id: number; data: ArrayBufferView; shape: number[]; version?: number; pyramid?: number }
	| { id: number; error: string; status?: number };

type OpenArray = zarr.Array<zarr.DataType, zarr.FetchStore>;

const arrays = new Map<string, Promise<OpenArray>>();
const running = new Map<number, AbortController>();
// The label zarr says which version of a chunk it served, by chunk URL: for
// the project's own chunks, the next edit's base version; for the coarser
// levels it makes, what the chunk was made from (see `derived`).
const versions = new Map<string, number>();
const pyramids = new Map<string, number>();

async function fetchNoting(request: Request): Promise<Response> {
	const response = await fetch(request);
	const version = response.headers.get("x-chunk-version");
	if (version !== null) versions.set(request.url, Number(version));
	const pyramid = response.headers.get("x-pyramid-version");
	if (pyramid !== null) pyramids.set(request.url, Number(pyramid));
	return response;
}

function openArray(url: string, path: string, derived: boolean): Promise<OpenArray> {
	const key = `${url}#${path}#${derived}`;
	let array = arrays.get(key);
	if (!array) {
		// Shard indexes are read with suffix ranges, which the gateway serves,
		// instead of a HEAD request first.
		const store = new zarr.FetchStore(url, {
			useSuffixRequest: true,
			fetch: shardedFetch(derived ? patientFetch(fetchNoting) : fetchNoting),
		});
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
		const derived = message.derived === true;
		const array = await openArray(message.url, message.path, derived);
		const [[z0, z1], [y0, y1], [x0, x1]] = message.region;
		const spatial = [zarr.slice(z0, z1), zarr.slice(y0, y1), zarr.slice(x0, x1)];
		const selection = message.channel === undefined ? spatial : [message.channel, ...spatial];
		const chunk = await zarr.get(array, selection, { opts: { signal: controller.signal } });
		const data = chunk.data as unknown as ArrayBufferView;
		// One chunk's region names that chunk (unsharded arrays only).
		const chunkUrl = `${message.url}${message.path}/c/${[z0, y0, x0].map((c) => Math.floor(c / 64)).join("/")}`;
		// A made chunk's version isn't an edit's base version, whatever else
		// the answer says.
		const version = derived ? undefined : versions.get(chunkUrl);
		const pyramid = derived ? pyramids.get(chunkUrl) : undefined;
		versions.delete(chunkUrl);
		pyramids.delete(chunkUrl);
		const reply: DecodeResponse = { id: message.id, data, shape: chunk.shape, version, pyramid };
		(self as unknown as Worker).postMessage(reply, [data.buffer as ArrayBuffer]);
	} catch (error) {
		const reply: DecodeResponse = { id: message.id, ...failure(error) };
		(self as unknown as Worker).postMessage(reply);
	} finally {
		running.delete(message.id);
	}
};
