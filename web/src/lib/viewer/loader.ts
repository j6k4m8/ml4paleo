/**
 * A pool of decode workers that loads one image's chunks (see
 * decode.worker.ts), for a ChunkStore.
 */

import type { Chunk, Loader } from "./chunks";
import type { DecodeResponse } from "./decode.worker";
import { CHUNK, type Level } from "./tiles";

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
		if ("error" in response) waiting.reject(new Error(response.error));
		else
			waiting.resolve({
				data: response.data as Chunk["data"],
				shape: response.shape,
			});
	}

	/** A Loader for `url` (an OME-Zarr image) whose ids are tile ids. */
	loader(url: string, levels: Level[]): Loader {
		return (id, signal) => {
			const [level, cz, cy, cx] = id.split("/").map(Number) as [number, number, number, number];
			const shape = levels[level]?.shape;
			if (!shape) return Promise.reject(new Error(`No level ${level}`));
			const region = [cz, cy, cx].map((c, axis) => [
				c * CHUNK,
				Math.min((c + 1) * CHUNK, shape[axis] ?? 0),
			]) as [[number, number], [number, number], [number, number]];
			const requestId = ++this.#id;
			const worker = this.#workers[this.#next++ % this.#workers.length];
			if (!worker) return Promise.reject(new Error("No decode workers"));
			return new Promise<Chunk>((resolve, reject) => {
				this.#waiting.set(requestId, { resolve, reject });
				signal.addEventListener("abort", () => {
					worker.postMessage({ type: "cancel", id: requestId });
					this.#waiting.delete(requestId);
					reject(new DOMException("No longer needed", "AbortError"));
				});
				worker.postMessage({ type: "load", id: requestId, url, level, region });
			});
		};
	}

	close(): void {
		for (const worker of this.#workers) worker.terminate();
	}
}
