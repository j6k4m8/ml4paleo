/**
 * A pool of decode workers (see decode.worker.ts), and loaders for the
 * ChunkStores of an image and of its labels.
 */

import type { Chunk, Loader } from "./chunks";
import type { ArrayRegion, DecodeResponse, Region } from "./decode.worker";
import { CHUNK, type Level, type Vec3 } from "./tiles";

/** A load that failed: what went wrong, and the HTTP status the server answered with, if it did. */
export class LoadError extends Error {
	constructor(
		message: string,
		readonly status?: number,
	) {
		super(message);
		this.name = "LoadError";
	}
}

export class WorkerPool {
	#workers: Worker[];
	#next = 0;
	#id = 0;
	#waiting = new Map<number, { resolve: (c: Chunk) => void; reject: (e: unknown) => void }>();

	constructor(size = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1))) {
		this.#workers = Array.from({ length: size }, () => {
			const worker = new Worker(new URL("./decode.worker.ts", import.meta.url), {
				type: "module",
			});
			worker.onmessage = (event: MessageEvent<DecodeResponse>) => this.#settle(event.data);
			return worker;
		});
	}

	#settle(response: DecodeResponse): void {
		const waiting = this.#waiting.get(response.id);
		if (!waiting) return;
		this.#waiting.delete(response.id);
		if ("error" in response) waiting.reject(new LoadError(response.error, response.status));
		else waiting.resolve({ data: response.data as Chunk["data"], shape: response.shape, version: response.version });
	}

	load(request: ArrayRegion, signal: AbortSignal): Promise<Chunk> {
		const id = ++this.#id;
		const worker = this.#workers[this.#next++ % this.#workers.length];
		if (!worker) return Promise.reject(new Error("No decode workers"));
		return new Promise<Chunk>((resolve, reject) => {
			const abort = () => {
				worker.postMessage({ type: "cancel", id });
				this.#waiting.delete(id);
				reject(new DOMException("No longer needed", "AbortError"));
			};
			signal.addEventListener("abort", abort, { once: true });
			const done = () => signal.removeEventListener("abort", abort);
			this.#waiting.set(id, {
				resolve: (chunk) => (done(), resolve(chunk)),
				reject: (error) => (done(), reject(error)),
			});
			worker.postMessage({ type: "load", id, ...request });
		});
	}

	close(): void {
		for (const worker of this.#workers) worker.terminate();
		for (const { reject } of this.#waiting.values()) reject(new DOMException("Closed", "AbortError"));
		this.#waiting.clear();
	}
}

/** The region of chunk (cz, cy, cx) in an array of `shape`. */
export function chunkRegion(key: Vec3, shape: Vec3): Region {
	return key.map((c, axis) => [c * CHUNK, Math.min((c + 1) * CHUNK, shape[axis] ?? 0)]) as Region;
}

function parse(id: string): number[] {
	return id.split("/").map(Number);
}

/** Loads image chunks, with ids `level/cz/cy/cx`, from an OME-Zarr image. */
export function imageLoader(pool: WorkerPool, url: string, levels: Level[]): Loader {
	return (id, signal) => {
		const [level = 0, cz = 0, cy = 0, cx = 0] = parse(id);
		const found = levels[level];
		if (!found) return Promise.reject(new Error(`No level ${level}`));
		const region = chunkRegion([cz, cy, cx], found.shape);
		return pool.load({ url, path: found.path, region, channel: 0 }, signal);
	};
}

/** Loads label chunks, with ids `cz/cy/cx`, from a project's label zarr. */
export function labelLoader(pool: WorkerPool, url: string, shape: Vec3): Loader {
	return (id, signal) => {
		const [cz = 0, cy = 0, cx = 0] = parse(id);
		return pool.load({ url, path: "class", region: chunkRegion([cz, cy, cx], shape) }, signal);
	};
}
