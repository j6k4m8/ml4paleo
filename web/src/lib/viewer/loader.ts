/**
 * A pool of decode workers (see decode.worker.ts), and loaders for the
 * ChunkStores of an image and of its labels.
 */

import { Busy, type Chunk, type Loader } from "./chunks";
import type { ArrayRegion, DecodeResponse, Region } from "./decoding";
import { retryDelay } from "./patient";
import { CHUNK, type Level, type Vec3 } from "./tiles";
import { LoadError } from "./load-errors";
export { LoadError } from "./load-errors";

// The worker can itself hang or crash, beyond fetch's 20-second deadline.
export const WORKER_TIMEOUT_MS = 45_000;

export class WorkerPool {
	#workers: (Worker | undefined)[];
	#next = 0;
	#id = 0;
	#closed = false;
	#waiting = new Map<number, { worker: Worker; resolve: (c: Chunk) => void; reject: (e: unknown) => void }>();

	constructor(size = Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1))) {
		this.#workers = Array.from({ length: size }, () => this.#create());
	}

	#create(): Worker {
		const worker = new Worker(new URL("./decode.worker.ts", import.meta.url), { type: "module" });
		worker.onmessage = (event: MessageEvent<DecodeResponse>) => this.#settle(worker, event.data);
		worker.onerror = (event) => {
			event.preventDefault();
			this.#discard(worker, new LoadError("The image reader stopped.", undefined, undefined, true));
		};
		worker.onmessageerror = () => this.#discard(worker, new LoadError("Couldn't read the image response.", undefined, undefined, true));
		return worker;
	}

	#discard(worker: Worker, error: LoadError): void {
		const index = this.#workers.indexOf(worker);
		if (index < 0) return; // A late event from a replaced worker.
		this.#workers[index] = undefined;
		worker.terminate();
		for (const [id, waiting] of this.#waiting) {
			if (waiting.worker !== worker) continue;
			this.#waiting.delete(id);
			waiting.reject(error);
		}
		// Recreate only on the next load, so a broken worker script cannot spin.
	}

	#settle(worker: Worker, response: DecodeResponse): void {
		const waiting = this.#waiting.get(response.id);
		if (!waiting || waiting.worker !== worker) return;
		this.#waiting.delete(response.id);
		if ("error" in response) waiting.reject(new LoadError(response.error, response.status, response.retryAfter, response.retryable));
		else {
			waiting.resolve({
				data: response.data as Chunk["data"],
				shape: response.shape,
				version: response.version,
				pyramid: response.pyramid,
			});
		}
	}

	load(request: ArrayRegion, signal: AbortSignal): Promise<Chunk> {
		if (this.#closed || signal.aborted) return Promise.reject(new DOMException("No longer needed", "AbortError"));
		if (!this.#workers.length) return Promise.reject(new Error("No decode workers"));
		const id = ++this.#id;
		const index = this.#next++ % this.#workers.length;
		let worker: Worker;
		try { worker = this.#workers[index] ??= this.#create(); }
		catch { return Promise.reject(new LoadError("Couldn't start the image reader.", undefined, undefined, true)); }
		return new Promise<Chunk>((resolve, reject) => {
			const timer = setTimeout(() => this.#discard(worker, new LoadError("Loading this area took too long.", undefined, undefined, true)), WORKER_TIMEOUT_MS);
			const done = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); };
			const abort = () => {
				this.#waiting.delete(id);
				done();
				reject(new DOMException("No longer needed", "AbortError"));
				try { worker.postMessage({ type: "cancel", id }); }
				catch { this.#discard(worker, new LoadError("The image reader stopped.", undefined, undefined, true)); }
			};
			signal.addEventListener("abort", abort, { once: true });
			this.#waiting.set(id, {
				worker,
				resolve: (chunk) => (done(), resolve(chunk)),
				reject: (error) => (done(), reject(error)),
			});
			try { worker.postMessage({ type: "load", id, ...request }); }
			catch { this.#discard(worker, new LoadError("Couldn't send work to the image reader.", undefined, undefined, true)); }
		});
	}

	close(): void {
		this.#closed = true;
		for (const worker of this.#workers) worker?.terminate();
		this.#workers.fill(undefined);
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

/**
 * Loads chunks of a project's label zarr at any of its levels (the image's
 * pyramid): ids `cz/cy/cx` for full resolution, `level/cz/cy/cx` for the
 * coarser levels, which the server makes when asked for. A coarser chunk the
 * server is still making (a 503) is `Busy`, for the store to ask again, after
 * the server's `Retry-After` and up to half as long again at random. A level
 * the server turns out not to have (it answers 404 for the array) is handed
 * to `missing`, and loads for it end quietly.
 */
export function labelLevelLoader(pool: WorkerPool, url: string, levels: () => Level[], missing: (level: number) => void): Loader {
	return async (id, signal) => {
		const parts = parse(id);
		const level = parts.length > 3 ? (parts[0] ?? 0) : 0;
		const [cz = 0, cy = 0, cx = 0] = parts.slice(-3);
		const found = levels()[level];
		if (!found) throw new DOMException("No such label level", "AbortError");
		try {
			const region = chunkRegion([cz, cy, cx], found.shape);
			return await pool.load({ url, path: found.path, region, derived: level > 0 }, signal);
		} catch (error) {
			if (level > 0 && error instanceof LoadError) {
				if (error.status === 404) {
					missing(level);
					throw new DOMException("No such label level", "AbortError");
				}
				if (error.status === 503) throw new Busy(retryDelay(error.retryAfter ?? null, Math.random()), error);
			}
			throw error;
		}
	};
}
