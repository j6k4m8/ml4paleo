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
	#classListeners = new Set<() => void>();
	#events: EventSource | null = null;
	// Edits sent but not yet confirmed, in order, by op: chunk id → delta.
	#local = new Map<string, Map<string, LocalDelta>>();
	// Chunks that reloaded while an op was unconfirmed, by op: its delta was
	// put back over whatever the server had, which may be newer.
	#reapplied = new Map<string, Set<string>>();
	/** Called if live updates stop for good (signed out, or removed from the project). */
	onStopped: (() => void) | null = null;

	constructor(
		private projectId: string,
		pool: WorkerPool,
		shape: Vec3,
	) {
		const url = absolute(`/api/projects/${projectId}/labels/zarr/`);
		this.store = new ChunkStore(labelLoader(pool, url, shape), CACHE_BYTES, 4);
		// Whatever the server sends, unconfirmed edits stay on screen.
		this.store.onLoad = (id, chunk) => {
			for (const [op, deltas] of this.#local) {
				const delta = deltas.get(id);
				if (!delta) continue;
				applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, delta.mask, delta.written, delta.onlyIf);
				this.#reapplied.get(op)?.add(id);
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
		this.#events.onerror = () => {
			// The browser retries dropped streams itself; a closed one is final.
			if (this.#events?.readyState === EventSource.CLOSED) this.onStopped?.();
		};
		this.#events.addEventListener("change", (event) => {
			const change = JSON.parse((event as MessageEvent<string>).data) as { chunks: { key: Vec3 }[] };
			this.reload(change.chunks.map(({ key }) => key.join("/")));
		});
	}

	/**
	 * Fetch chunks again. Ones a view shows reload in the background (what's
	 * on screen stays until the new copy arrives); others are dropped, along
	 * with any load in flight, so the next look fetches them afresh.
	 */
	reload(ids: string[]): void {
		const dropped: string[] = [];
		for (const id of ids) {
			if (this.store.get(id) && this.store.isWanted(id)) {
				this.store.refresh(id).then(
					() => this.#emit([id]),
					() => {},
				);
			} else {
				this.store.invalidate(id);
				dropped.push(id);
			}
		}
		this.#emit(dropped);
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
		this.#reapplied.set(op, new Set());
		this.#emit(changed);
	}

	/**
	 * The server answered for an op: keep its result (taking the versions it
	 * made) or, if it was refused, fetch its chunks again to undo the preview.
	 */
	settle(op: string, versions: { key: Vec3; version: number }[] | null): void {
		const local = this.#local.get(op);
		const reapplied = this.#reapplied.get(op) ?? new Set<string>();
		this.#local.delete(op);
		this.#reapplied.delete(op);
		for (const id of local?.keys() ?? []) this.store.unpin(id);
		if (versions) {
			this.noteVersions(versions);
			if (reapplied.size > 0) this.reload([...reapplied]);
		} else if (local) {
			this.reload([...local.keys()]);
		}
	}

	/**
	 * Take the versions an op of this page made. A chunk this page had at the
	 * version just before is now current; otherwise someone else changed it
	 * too, and it reloads.
	 */
	noteVersions(versions: { key: Vec3; version: number }[]): void {
		const stale: string[] = [];
		for (const { key, version } of versions) {
			const id = key.join("/");
			const chunk = this.store.get(id);
			if (!chunk || chunk.version === undefined) continue;
			if (chunk.version === version - 1) chunk.version = version;
			else if (chunk.version < version) stale.push(id);
		}
		if (stale.length > 0) this.reload(stale);
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

	/**
	 * Add a class to the project. It gets the next value never used, and the
	 * views learn its color.
	 */
	async addClass(name: string, color: string): Promise<LabelClass> {
		const made = await api<LabelClass>(`/api/projects/${this.projectId}/labels/classes`, { body: { name, color } });
		// A new class's value is the highest ever used, so it goes last.
		this.classes = [...this.classes, made];
		for (const listener of this.#classListeners) listener();
		return made;
	}

	/** Call `listener` when a class is added here (its color needs drawing). */
	onClasses(listener: () => void): () => void {
		this.#classListeners.add(listener);
		return () => this.#classListeners.delete(listener);
	}

	stop(): void {
		this.#events?.close();
		this.store.keepOnly(new Set());
		this.#listeners.clear();
		this.#classListeners.clear();
	}
}
