/**
 * A project's labels as a viewer layer: the classes and their colors, a
 * cache of label chunks, live updates when anyone edits them, and this
 * page's own edits shown before the server confirms them.
 */

import { api } from "#lib/api.ts";
import { applyLocally, type DeltaIn, decodeDelta } from "../labels/deltas";
import { ChunkStore } from "./chunks";
import { absolute } from "./image";
import { LoadError, labelLoader, type WorkerPool } from "./loader";
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
	 * Once the server has applied the op, the chunk version it made. A copy
	 * read at that version or later has the edit. One read just before it
	 * gets the edit put back, which makes it that version. An older one
	 * (someone else changed the chunk too) or one whose version isn't known
	 * gets it put back and the chunk loads once more. Such a delta lasts
	 * only as long as the load of its chunk that may predate the op: any
	 * later load has the edit.
	 */
	made?: number;
	/** A copy that needed it put back and wasn't made current by it arrived, and the chunk was sent for once more. */
	again?: boolean;
}

const CACHE_BYTES = 128 * 1024 * 1024;
// How many of the chunks this page edited last it remembers (`recent`).
const RECENT = 256;
// A reload that failed is tried again after this long, doubling each time up
// to the longest, each wait 25% shorter or longer at random so chunks that
// failed together don't all try again together.
const RETRY_MS = 1000;
const RETRY_LONGEST_MS = 30_000;
const RETRY_SPREAD = 0.25;
// Failing this many times in a row, other than for the network, a chunk isn't tried again.
const FAILURES = 8;

/**
 * Whether a reload that failed is worth trying again, `failures` failures in
 * a row in. A network failure is: it ends when the network comes back. A
 * server error, or a failure to read what it sent, is for a while. Being
 * signed out or refused isn't, and neither is any other answer.
 */
function worthRetrying(error: unknown, failures: number): boolean {
	const status = error instanceof LoadError ? error.status : undefined;
	if (status === undefined) {
		// What the browser throws when fetch gets no answer is a TypeError.
		const network = error instanceof Error && error.message.startsWith("TypeError");
		return network || failures < FAILURES;
	}
	if (status >= 500 || status === 408 || status === 425 || status === 429) return failures < FAILURES;
	return false;
}

/**
 * An edit as it goes out. A strict one is refused if its chunks changed since
 * the page read them, so it carries the versions the page's copies stand
 * for; where any isn't known, it goes out as a plain edit instead.
 */
export function strictOn<T extends { strict: boolean; deltas: DeltaIn[] }>(layer: Pick<LabelLayer, "versionOf">, op: T): T {
	if (!op.strict) return op;
	const versions = op.deltas.map((d) => layer.versionOf(d.key.join("/")));
	if (versions.some((v) => v === undefined)) return { ...op, strict: false };
	return { ...op, deltas: op.deltas.map((d, i) => ({ ...d, base_version: versions[i]! })) };
}

export class LabelLayer {
	store: ChunkStore;
	classes: LabelClass[] = [];
	#listeners = new Set<(ids: string[]) => void>();
	#classListeners = new Set<() => void>();
	#events: EventSource | null = null;
	// This page's edits that a copy of their chunk may not show yet, by op,
	// in order: chunk id → delta. Until the server answers, an op's deltas
	// go over every copy that loads; after, only over copies older than the
	// version the op made.
	#local = new Map<string, Map<string, LocalDelta>>();
	// Chunks this page edited, the latest last.
	#recent = new Set<string>();
	// Reloads that failed and will be tried again: how many times each has
	// failed in a row, and the timer of the next try.
	#failures = new Map<string, number>();
	#retries = new Map<string, ReturnType<typeof setTimeout>>();
	#stopped = false;
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
				if (delta.made !== undefined && chunk.version !== undefined && chunk.version >= delta.made) {
					// Read after the op: it has the edit.
					this.#forget(op, id);
					continue;
				}
				applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, delta.mask, delta.written, delta.onlyIf);
				if (delta.made === undefined) continue;
				// A copy from before the edit (its load started first).
				if (chunk.version === delta.made - 1) {
					// Just before it: with the edit put back it is what the server has.
					chunk.version = delta.made;
					this.#forget(op, id);
				} else if (delta.again) {
					this.#forget(op, id);
				} else {
					// Someone else changed the chunk too, or its version isn't known:
					// show the edit on it, and load it once more.
					delta.again = true;
					stale = true;
				}
			}
			if (stale) queueMicrotask(() => this.#reloadFor([id]));
		};
	}

	/** Chunks this page edited (ids, `cz/cy/cx`), the latest last. */
	get recent(): string[] {
		return [...this.#recent];
	}

	#forget(op: string, id: string): void {
		const deltas = this.#local.get(op);
		deltas?.delete(id);
		if (deltas?.size === 0) this.#local.delete(op);
	}

	/** This page's confirmed edits still waiting on chunks, by op: chunk id → delta. */
	*#confirmed(ids: string[]): Generator<[string, string, LocalDelta]> {
		for (const [op, deltas] of this.#local) {
			for (const id of ids) {
				const delta = deltas.get(id);
				if (delta?.made !== undefined) yield [op, id, delta];
			}
		}
	}

	/** Load chunks again for the confirmed edits still waiting on them. */
	#reloadFor(ids: string[]): void {
		const versions = new Map<string, number>();
		for (const [, id, delta] of this.#confirmed(ids)) versions.set(id, Math.max(delta.made!, versions.get(id) ?? 0));
		this.reload(ids, versions);
		this.#tie(ids);
	}

	/**
	 * Confirmed edits on these chunks last as long as the load of each that's
	 * under way now, or end now if none is: a load started later has them.
	 */
	#tie(ids: string[]): void {
		for (const [op, id, delta] of this.#confirmed(ids)) {
			const loading = this.store.loading(id);
			if (!loading) {
				this.#forget(op, id);
				continue;
			}
			loading.catch(() => {
				if (this.#local.get(op)?.get(id) === delta) this.#forget(op, id);
			});
		}
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
			// The browser retries dropped streams itself; a closed one is final,
			// and what it means (signed out, or removed from the project) means
			// reloads that fail won't work when tried again either.
			if (this.#events?.readyState === EventSource.CLOSED) {
				this.#stopRetrying();
				this.onStopped?.();
			}
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
			const id = key.join("/");
			const chunk = this.store.peek(id);
			if (chunk?.version !== undefined && chunk.version >= version) return false;
			// Already loading again for this version or a later one (an undo's
			// answer and its change event both name it).
			return (this.store.refreshing(id) ?? -1) < version;
		});
		this.reload(
			stale.map(({ key }) => key.join("/")),
			new Map(stale.map(({ key, version }) => [key.join("/"), version])),
		);
	}

	/**
	 * Fetch chunks again. Ones a view shows reload in the background (what's
	 * on screen stays until the new copy arrives); others are dropped, along
	 * with any load in flight, so the next look fetches them afresh.
	 * `versions` says what each reload is for (the version the new copy will
	 * have at least), so a change event for that version doesn't restart it.
	 */
	reload(ids: string[], versions?: Map<string, number>): void {
		const dropped: string[] = [];
		for (const id of ids) {
			if (this.store.get(id) && this.store.isWanted(id)) {
				this.store.refresh(id, versions?.get(id)).then(
					() => {
						this.#giveUp(id);
						this.#emit([id]);
					},
					(error: unknown) => {
						// A reload that failed leaves the copy shown, and is tried again.
						if (!(error instanceof DOMException && error.name === "AbortError")) {
							this.#retry(id, versions?.get(id), error);
							return;
						}
						// One stopped (no view wants the chunk now) took the out-of-date
						// copy with it, so views forget what they drew from it and load
						// the chunk afresh, unless another reload took over, which says
						// so itself.
						this.#giveUp(id);
						if (!this.store.isLoading(id)) this.#emit([id]);
					},
				);
			} else {
				this.store.invalidate(id);
				this.#giveUp(id);
				dropped.push(id);
			}
		}
		this.#emit(dropped);
	}

	/**
	 * Load a chunk again after a failed reload (the `error`), if that's worth
	 * doing and the copy shown is still the out-of-date one.
	 */
	#retry(id: string, version: number | undefined, error: unknown): void {
		clearTimeout(this.#retries.get(id));
		const stale = this.store.peek(id);
		const failures = (this.#failures.get(id) ?? 0) + 1;
		if (this.#stopped || !stale || !worthRetrying(error, failures)) return this.#giveUp(id);
		this.#failures.set(id, failures);
		const timer = setTimeout(() => {
			// A newer copy came since, or the chunk is loading again, or it's gone.
			if (this.store.peek(id) !== stale || this.store.isLoading(id)) return this.#giveUp(id);
			this.#retries.delete(id);
			this.reload([id], version === undefined ? undefined : new Map([[id, version]]));
		}, Math.min(RETRY_LONGEST_MS, RETRY_MS * 2 ** (failures - 1)) * (1 - RETRY_SPREAD + 2 * RETRY_SPREAD * Math.random()));
		this.#retries.set(id, timer);
	}

	#giveUp(id: string): void {
		clearTimeout(this.#retries.get(id));
		this.#retries.delete(id);
		this.#failures.delete(id);
	}

	/**
	 * The server's chunk version that the copy shown stands for (this
	 * page's own edits on it counted in), if it has the chunk: the version
	 * a strict edit goes out on. Where someone else changed the chunk
	 * between, it is the version before, so the edit is refused.
	 */
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
			this.#recent.delete(id);
			this.#recent.add(id);
			const chunk = this.store.get(id);
			// Without a copy here, views drop anything drawn from an old one and
			// load the chunk again, which shows the edit.
			if (!chunk || applyLocally(chunk.data as Uint8Array, chunk.shape, delta.box, mask, written, delta.only_if) > 0) {
				changed.push(id);
			}
		}
		for (const id of this.#recent) {
			if (this.#recent.size <= RECENT) break;
			this.#recent.delete(id);
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
			// Chunks already loading come clean: the refused edit is out of `#local`.
			this.reload([...local.keys()].filter((id) => !this.store.isLoading(id)));
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
			delta.made = version;
			const chunk = this.store.peek(id);
			if (chunk?.version === undefined) {
				// A copy of a version not known: load it again, showing the edit meanwhile.
				if (chunk) again.push(id);
			} else if (chunk.version === version - 1) {
				chunk.version = version;
			} else if (chunk.version < version - 1 || chunk.version > version) {
				// Someone else changed the chunk too, or a copy newer than the
				// op was read and the edit went over it again: load it again.
				// Put back over later changes, the edit hides them, so the copy
				// no longer stands for their version: it says the op's, which
				// the server is past, and strict edits on it are refused until
				// the new copy lands.
				if (chunk.version > version) chunk.version = version;
				again.push(id);
			}
		}
		if (local.size === 0) this.#local.delete(op);
		this.reload(again, made);
		// What's left waits for the loads of its chunks that may predate it.
		this.#tie([...local.keys()]);
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
		// Along with whatever others added meanwhile; but the new one is ours either way.
		await this.refreshClasses().catch(() => {});
		if (!this.classes.some((c) => c.value === made.value)) {
			// A new class's value is the highest ever used, so it goes last.
			this.classes = [...this.classes, made];
			this.#classesChanged();
		}
		return made;
	}

	/** Read the project's classes again, in case someone added some. Views hear of any change. */
	async refreshClasses(): Promise<void> {
		const found = await api<LabelClass[]>(`/api/projects/${this.projectId}/labels/classes`);
		if (JSON.stringify(found) === JSON.stringify(this.classes)) return;
		this.classes = found;
		this.#classesChanged();
	}

	#classesChanged(): void {
		for (const listener of this.#classListeners) {
			// One view's trouble mustn't keep the rest from hearing.
			try {
				listener();
			} catch (e) {
				console.error(e);
			}
		}
	}

	/** Call `listener` when a class is added here (its color needs drawing). */
	onClasses(listener: () => void): () => void {
		this.#classListeners.add(listener);
		return () => this.#classListeners.delete(listener);
	}

	#stopRetrying(): void {
		this.#stopped = true;
		for (const timer of this.#retries.values()) clearTimeout(timer);
		this.#retries.clear();
		this.#failures.clear();
	}

	stop(): void {
		this.#stopRetrying();
		this.#events?.close();
		this.store.keepOnly(new Set());
		this.#listeners.clear();
		this.#classListeners.clear();
	}
}
