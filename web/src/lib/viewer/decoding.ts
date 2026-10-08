/**
 * What a decode worker does with a request: read a region of a zarr array
 * and say what it read, or why it couldn't. It lives here, not in the
 * worker's file (which starts working as soon as it is loaded), so it can be
 * tested.
 */

import * as zarr from "zarrita";
import { failure, shardedFetch } from "./image";
import { CHUNK } from "./tiles";

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
	 * label pyramid): a chunk's version is the `X-Pyramid-Version` it was
	 * made from, not an edit's `X-Chunk-Version`.
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
	// For a 503, the answer's `Retry-After`, as the server wrote it.
	| { id: number; error: string; status?: number; retryAfter?: string };

type OpenArray = zarr.Array<zarr.DataType, zarr.FetchStore>;

/** The URL of the chunk a region starts in, which is the region itself for a chunk's own (unsharded arrays only). */
export function chunkUrl({ url, path, region }: ArrayRegion): string {
	const [[z], [y], [x]] = region;
	return `${url}${path}/c/${[z, y, x].map((c) => Math.floor(c / CHUNK)).join("/")}`;
}

/**
 * A function answering requests, which reads arrays through `fetcher`, and
 * remembers the arrays it opened. The answer is the reply and what to
 * transfer with it. A 503 isn't asked again here: the reply says so, with its
 * `Retry-After`, and the page's chunk store asks again without a place of
 * its own taken by the wait.
 */
export function createDecoder(fetcher: (request: Request) => Promise<Response> = (request) => fetch(request)) {
	const arrays = new Map<string, Promise<OpenArray>>();
	// What the server said of each chunk it sent, by chunk URL: for the
	// project's own chunks, the version, which is the next edit's base
	// version; for the coarser levels it makes (see `derived`), what the
	// chunk was made from; and a 503's `Retry-After`.
	const versions = new Map<string, number>();
	const pyramids = new Map<string, number>();
	const retries = new Map<string, string>();

	async function fetchNoting(request: Request): Promise<Response> {
		const response = await fetcher(request);
		const version = response.headers.get("x-chunk-version");
		if (version !== null) versions.set(request.url, Number(version));
		const pyramid = response.headers.get("x-pyramid-version");
		if (pyramid !== null) pyramids.set(request.url, Number(pyramid));
		const retry = response.headers.get("retry-after");
		if (response.status === 503 && retry !== null) retries.set(request.url, retry);
		return response;
	}

	function openArray(url: string, path: string): Promise<OpenArray> {
		const key = `${url}#${path}`;
		let array = arrays.get(key);
		if (!array) {
			// Shard indexes are read with suffix ranges, which the gateway serves,
			// instead of a HEAD request first.
			const store = new zarr.FetchStore(url, { useSuffixRequest: true, fetch: shardedFetch(fetchNoting) });
			array = zarr.open.v3(zarr.root(store).resolve(path), { kind: "array" });
			// Let a later request try again after a failure.
			array.catch(() => arrays.delete(key));
			arrays.set(key, array);
		}
		return array;
	}

	return async function decode(message: DecodeRequest, signal: AbortSignal): Promise<{ reply: DecodeResponse; transfer: ArrayBuffer[] }> {
		const name = chunkUrl(message);
		try {
			const array = await openArray(message.url, message.path);
			const [[z0, z1], [y0, y1], [x0, x1]] = message.region;
			const spatial = [zarr.slice(z0, z1), zarr.slice(y0, y1), zarr.slice(x0, x1)];
			const selection = message.channel === undefined ? spatial : [message.channel, ...spatial];
			const chunk = await zarr.get(array, selection, { opts: { signal } });
			const data = chunk.data as unknown as ArrayBufferView;
			// A made chunk's version isn't an edit's base version, whatever else
			// the answer says.
			const derived = message.derived === true;
			const version = derived ? undefined : versions.get(name);
			const pyramid = derived ? pyramids.get(name) : undefined;
			versions.delete(name);
			pyramids.delete(name);
			return { reply: { id: message.id, data, shape: chunk.shape, version, pyramid }, transfer: [data.buffer as ArrayBuffer] };
		} catch (error) {
			const retryAfter = retries.get(name);
			retries.delete(name);
			return { reply: { id: message.id, ...failure(error), ...(retryAfter === undefined ? {} : { retryAfter }) }, transfer: [] };
		}
	};
}
