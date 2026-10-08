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
	/**
	 * Once the server has applied the op, the chunk version it made: copies
	 * from then on have the edit, and only older ones need it put back.
	 */
	made?: number;
	/** A copy older than `made` arrived and was sent for again. */
	again?: boolean;
}

const CACHE_BYTES = 128 * 1024 * 1024;

export class LabelLayer {
	store: ChunkStore;
	classes: LabelClass[] = [];
	#listeners = new Set<(ids: string[]) => void>();
	#events: EventSource | null = null;
	// This page's edits that a copy of their chunk may not show yet, by op,
	// in order: chunk id → delta. Until the server answers, an op's deltas
	// go over every copy that loads; after, only over copies older than the
	// version the op made.
	#local = new Map<string, Map<string, LocalDelta>>();
	/** Called if live updates stop for good (signed out, or removed from the project). */
	onStopped: (() => void) | null = null;

	constructor(
		private projectId: string,
		pool: WorkerPool,
		shape: Vec3,
	) {
		const url = absolute(`/api/projects/${projectId}/labels/zarr/`);
		this.store = new ChunkStore(labelLoader(pool, url, shape), CACHE_BYTES, 4);
		// Whatever the server sends, this page's edits stay on screen.
		this.store.onLoad = (id, chunk) => {
			let stale = false;
			for (const [op, deltas] of this.#local) {
				const delta = deltas.get(id);
				if (!delta) continue;
				if (delta.made !== undefined) {
					if (chunk.version === undefined || chunk.version >= delta.made) {
						this.#forget(op, id);
						continue;
					}
					// A copy from before the edit (its load started first): show the
					// edit on it, and load it once more.
					stale ||= !delta.again;
					delta.again = true;
				}
				applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, delta.mask, delta.written, delta.onlyIf);
			}
			if (stale) queueMicrotask(() => this.reload([id]));
		};
	}

	#forget(op: string, id: string): void {
		const deltas = this.#local.get(op);
		deltas?.delete(id);
		if (deltas?.size === 0) this.#local.delete(op);
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
			const change = JSON.parse((event as MessageEvent<string>).data) as { chunks: { key: Vec3; version: number }[] };
			this.changed(change.chunks);
		});
	}

	/**
	 * Chunks that changed on the server, at these versions: anyone's edits,
	 * or this page's undos and redos (whose results it can't work out). The
	 * ones this page holds an older copy of, or is still loading, load again.
	 */
	changed(chunks: { key: Vec3; version: number }[]): void {
		const stale = chunks.filter(({ key, version }) => {
			const chunk = this.store.peek(key.join("/"));
			return chunk?.version === undefined || chunk.version < version;
		});
		this.reload(stale.map(({ key }) => key.join("/")));
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

	/** Show an op's deltas at once, and keep showing them until its copies arrive. */
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
	 * The server answered for an op, with the chunk versions it made, or
	 * null if it refused the op (its chunks load again to take it back).
	 * A copy this page had at the version just before is the server's copy
	 * now. A copy someone else changed meanwhile, or one still loading that
	 * may predate the op, keeps showing the edit until a copy with it comes.
	 */
	settle(op: string, versions: { key: Vec3; version: number }[] | null): void {
		const local = this.#local.get(op);
		if (!local) return;
		for (const id of local.keys()) this.store.unpin(id);
		if (!versions) {
			this.#local.delete(op);
			this.reload([...local.keys()]);
			return;
		}
		const made = new Map(versions.map(({ key, version }) => [key.join("/"), version]));
		const again: string[] = [];
		for (const [id, delta] of local) {
			const version = made.get(id);
			if (version === undefined) {
				local.delete(id);
				continue;
			}
			const chunk = this.store.peek(id);
			let keep = this.store.isLoading(id);
			if (chunk?.version !== undefined) {
				if (chunk.version === version - 1) {
					chunk.version = version;
				} else if (chunk.version < version - 1) {
					keep = true;
					again.push(id);
				} else if (chunk.version > version) {
					// A copy newer than the op, which went over it again.
					again.push(id);
				}
			}
			if (keep) delta.made = version;
			else local.delete(id);
		}
		if (local.size === 0) this.#local.delete(op);
		this.reload(again);
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
