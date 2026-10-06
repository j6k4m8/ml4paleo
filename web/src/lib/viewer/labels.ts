/**
 * A project's labels as a viewer layer: the classes and their colors, a
 * cache of label chunks, and live updates when anyone edits them.
 */

import { api } from "#lib/api.ts";
import { ChunkStore } from "./chunks";
import { absolute } from "./image";
import { labelLoader, type WorkerPool } from "./loader";
import type { Vec3 } from "./tiles";

export interface LabelClass {
	value: number;
	name: string;
	color: string;
}

const CACHE_BYTES = 128 * 1024 * 1024;

export class LabelLayer {
	store: ChunkStore;
	classes: LabelClass[] = [];
	#listeners = new Set<(ids: string[]) => void>();
	#events: EventSource | null = null;
	/** Called if live updates stop for good (signed out, or removed from the project). */
	onStopped: (() => void) | null = null;

	constructor(
		private projectId: string,
		pool: WorkerPool,
		shape: Vec3,
	) {
		const url = absolute(`/api/projects/${projectId}/labels/zarr/`);
		this.store = new ChunkStore(labelLoader(pool, url, shape), CACHE_BYTES, 4);
	}

	/** Colors by label value (background, 1, has none). */
	get colors(): Map<number, string> {
		return new Map(this.classes.map((c) => [c.value, c.color]));
	}

	async start(): Promise<void> {
		const base = `/api/projects/${this.projectId}/labels`;
		this.classes = await api<LabelClass[]>(`${base}/classes`);
		const [latest] = await api<{ seq: number }[]>(`${base}/ops?limit=1`);
		this.#events = new EventSource(`${base}/events?after=${latest?.seq ?? 0}`);
		this.#events.onerror = () => {
			// The browser retries dropped streams itself; a closed one is final.
			if (this.#events?.readyState === EventSource.CLOSED) this.onStopped?.();
		};
		this.#events.addEventListener("change", (event) => {
			const change = JSON.parse((event as MessageEvent<string>).data) as {
				chunks: { key: Vec3 }[];
			};
			const ids = change.chunks.map(({ key }) => key.join("/"));
			for (const id of ids) this.store.invalidate(id);
			for (const listener of this.#listeners) listener(ids);
		});
	}

	/** Call `listener` with the ids of chunks that changed. */
	onChange(listener: (ids: string[]) => void): () => void {
		this.#listeners.add(listener);
		return () => this.#listeners.delete(listener);
	}

	stop(): void {
		this.#events?.close();
		this.store.keepOnly(new Set());
		this.#listeners.clear();
	}
}
