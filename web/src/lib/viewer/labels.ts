/**
 * A project's labels as a viewer layer: the classes and their colors, a
 * cache of label chunks, live updates when anyone edits them, and this
 * page's own edits shown before the server confirms them.
 */

import { api } from "#lib/api.ts";
import { applyLocally, type DeltaIn, decodeDelta } from "../labels/deltas";
import { ChunkStore } from "./chunks";
import { absolute } from "./image";
import { labelLoader, type WorkerPool } from "./loader";
import type { Vec3 } from "./tiles";

export interface LabelClass {
	value: number;
	name: string;
	color: string;
}

interface LocalDelta {
	box: DeltaIn["box"];
	mask: Uint8Array;
	written: Uint8Array | number;
	onlyIf: string;
}

const CACHE_BYTES = 128 * 1024 * 1024;

export class LabelLayer {
	store: ChunkStore;
	classes: LabelClass[] = [];
	#listeners = new Set<(ids: string[]) => void>();
	#events: EventSource | null = null;
	// Edits sent but not yet confirmed, in order, by op: chunk id → delta.
	#local = new Map<string, Map<string, LocalDelta>>();

	constructor(
		private projectId: string,
		pool: WorkerPool,
		shape: Vec3,
	) {
		const url = absolute(`/api/projects/${projectId}/labels/zarr/`);
		this.store = new ChunkStore(labelLoader(pool, url, shape), CACHE_BYTES, 4);
		// Whatever the server sends, unconfirmed edits stay on screen.
		this.store.onLoad = (id, chunk) => {
			for (const deltas of this.#local.values()) {
				const delta = deltas.get(id);
				if (delta) applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, delta.mask, delta.written, delta.onlyIf);
			}
		};
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
		this.#events.addEventListener("change", (event) => {
			const change = JSON.parse((event as MessageEvent<string>).data) as { chunks: { key: Vec3 }[] };
			this.reload(change.chunks.map(({ key }) => key.join("/")));
		});
	}

	/** Fetch chunks again (keeping what's on screen meanwhile), then redraw them. */
	reload(ids: string[]): void {
		for (const id of ids) {
			if (this.store.get(id)) {
				this.store.refresh(id).then(
					() => this.#emit([id]),
					() => {},
				);
			} else {
				this.#emit([id]);
			}
		}
	}

	/** The chunk version this page last read, if it has the chunk. */
	versionOf(id: string): number | undefined {
		return this.store.get(id)?.version;
	}

	/** Show an op's deltas at once, until `settle` is called for it. */
	applyLocal(op: string, deltas: DeltaIn[]): void {
		const local = new Map<string, LocalDelta>();
		const changed: string[] = [];
		for (const delta of deltas) {
			const id = delta.key.join("/");
			const { mask, written } = decodeDelta(delta);
			local.set(id, { box: delta.box, mask, written, onlyIf: delta.only_if });
			this.store.pin(id);
			const chunk = this.store.get(id);
			if (chunk && applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, mask, written, delta.only_if) > 0) {
				changed.push(id);
			}
		}
		this.#local.set(op, local);
		this.#emit(changed);
	}

	/**
	 * The server answered for an op: keep its result (taking the versions it
	 * made) or, if it was refused, fetch its chunks again to undo the preview.
	 */
	settle(op: string, versions: { key: Vec3; version: number }[] | null): void {
		const local = this.#local.get(op);
		this.#local.delete(op);
		for (const id of local?.keys() ?? []) this.store.unpin(id);
		if (versions) {
			for (const { key, version } of versions) {
				const chunk = this.store.get(key.join("/"));
				if (chunk && (chunk.version ?? -1) < version) chunk.version = version;
			}
		} else if (local) {
			this.reload([...local.keys()]);
		}
	}

	#emit(ids: string[]): void {
		if (ids.length === 0) return;
		for (const listener of this.#listeners) listener(ids);
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
