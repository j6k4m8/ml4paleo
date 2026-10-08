/**
 * A cache of decoded chunks, kept under a byte budget (least recently used
 * chunks go first), that loads each chunk once however often it is asked
 * for, runs a limited number of loads at a time, most wanted first, and
 * drops loads that no view needs any more.
 */

export interface Chunk {
	data: ArrayBufferView & { length: number };
	/** Shape of the chunk, (z, y, x). Edge chunks may be smaller. */
	shape: number[];
	/** For label chunks, the version the server served. */
	version?: number;
}

export type Loader = (id: string, signal: AbortSignal) => Promise<Chunk>;

interface Pending {
	id: string;
	promise: Promise<Chunk>;
	resolve: (chunk: Chunk) => void;
	reject: (error: unknown) => void;
	controller: AbortController;
	/** Replaces a cached copy that is now out of date. */
	refresh: boolean;
}

// Where a chunk nobody wants waits: after every wanted one.
const UNRANKED = Number.MAX_SAFE_INTEGER;

export class ChunkStore {
	#cache = new Map<string, Chunk>();
	#bytes = 0;
	#pending = new Map<string, Pending>();
	#queue: Pending[] = [];
	#sorted = true;
	#running = 0;
	#pinned = new Set<string>();
	#wanted = new Map<string, string[]>();
	#protected = new Map<string, Set<string>>();
	// Each wanted chunk's best place in any owner's list: the queue's order.
	#rank = new Map<string, number>();
	/** Called with each loaded chunk before it is cached and returned. */
	onLoad: ((id: string, chunk: Chunk) => void) | null = null;

	constructor(
		private load: Loader,
		private maxBytes: number,
		private concurrency = 8,
	) {}

	get bytes(): number {
		return this.#bytes;
	}

	/** A cached chunk, marked as just used. */
	get(id: string): Chunk | undefined {
		const chunk = this.#cache.get(id);
		if (chunk) {
			this.#cache.delete(id);
			this.#cache.set(id, chunk);
		}
		return chunk;
	}

	/** A cached chunk, leaving its place in line for eviction alone. */
	peek(id: string): Chunk | undefined {
		return this.#cache.get(id);
	}

	/** Whether a load of the chunk is queued or running. */
	isLoading(id: string): boolean {
		return this.#pending.has(id);
	}

	/** Load a chunk (once), queueing it behind chunks wanted more. */
	request(id: string): Promise<Chunk> {
		const cached = this.get(id);
		if (cached) return Promise.resolve(cached);
		const pending = this.#pending.get(id);
		if (pending) return pending.promise;
		return this.#enqueue(id, false);
	}

	/**
	 * Load a chunk again (for example after someone edited it), keeping the
	 * cached copy in use until the new one replaces it.
	 */
	refresh(id: string): Promise<Chunk> {
		this.#cancel(id, "Changed while loading", false);
		return this.#enqueue(id, true);
	}

	/** Whether any view needs this chunk now. */
	isWanted(id: string): boolean {
		return this.#rank.has(id);
	}

	#enqueue(id: string, refresh: boolean): Promise<Chunk> {
		let resolve!: (chunk: Chunk) => void;
		let reject!: (error: unknown) => void;
		const promise = new Promise<Chunk>((ok, fail) => {
			resolve = ok;
			reject = fail;
		});
		const entry: Pending = { id, promise, resolve, reject, controller: new AbortController(), refresh };
		this.#pending.set(id, entry);
		this.#queue.push(entry);
		this.#sorted = false;
		this.#pump();
		return promise;
	}

	/**
	 * Say which chunks `owner` (for example one of several views sharing
	 * this store) needs now, most wanted first; loads no owner needs are
	 * cancelled. Loads run in order of each chunk's best place in any
	 * owner's list, so views sharing the store take turns. Of the chunks
	 * wanted, the `shown` ones (by default all) are never evicted, even over
	 * budget, and all count as just used.
	 */
	want(owner: string, ids: Iterable<string>, shown?: Iterable<string>): void {
		const list = [...ids];
		if (list.length > 0) this.#wanted.set(owner, list);
		else this.#wanted.delete(owner);
		const kept = new Set(shown ?? list);
		if (kept.size > 0) this.#protected.set(owner, kept);
		else this.#protected.delete(owner);
		this.#rank = new Map();
		for (const wanted of this.#wanted.values()) {
			wanted.forEach((id, place) => {
				if (place < (this.#rank.get(id) ?? UNRANKED)) this.#rank.set(id, place);
			});
		}
		// The most wanted last, so it is the last to go.
		for (let i = list.length - 1; i >= 0; i--) this.get(list[i]!);
		this.#sorted = false;
		this.keepOnly(new Set(this.#rank.keys()));
	}

	#isShown(id: string): boolean {
		for (const set of this.#protected.values()) if (set.has(id)) return true;
		return false;
	}

	/** Cancel queued and running loads of chunks not in `wanted`. */
	keepOnly(wanted: Set<string>): void {
		const cancelled = [...this.#pending.values()].filter((entry) => !wanted.has(entry.id));
		if (cancelled.length === 0) return;
		const gone = new Set(cancelled);
		this.#queue = this.#queue.filter((queued) => !gone.has(queued));
		for (const entry of cancelled) this.#stop(entry, "No longer needed", true);
	}

	#cancel(id: string, reason: string, dropStale = true): void {
		const entry = this.#pending.get(id);
		if (!entry) return;
		this.#queue = this.#queue.filter((queued) => queued !== entry);
		this.#stop(entry, reason, dropStale);
	}

	/**
	 * Stop a queued or running load (already out of the queue). A refresh
	 * stopped for good leaves an out-of-date copy, which goes too (unless
	 * another refresh follows), so the next request loads afresh.
	 */
	#stop(entry: Pending, reason: string, dropStale: boolean): void {
		entry.controller.abort();
		this.#pending.delete(entry.id);
		if (entry.refresh && dropStale) this.#drop(entry.id);
		entry.reject(new DOMException(reason, "AbortError"));
	}

	/** Keep a chunk however full the cache gets (for example while edited). */
	pin(id: string): void {
		this.#pinned.add(id);
	}

	unpin(id: string): void {
		this.#pinned.delete(id);
		this.#evict();
	}

	/**
	 * Forget a chunk, for example after an edit changed it. A load already
	 * running might return the old contents, so it is cancelled too.
	 */
	invalidate(id: string): void {
		this.#drop(id);
		this.#cancel(id, "Changed while loading");
	}

	#drop(id: string): void {
		const chunk = this.#cache.get(id);
		if (chunk) {
			this.#cache.delete(id);
			this.#bytes -= chunk.data.byteLength;
		}
	}

	/** The queued load to start next: the most wanted. */
	#next(): Pending | undefined {
		if (!this.#sorted) {
			const rank = (entry: Pending) => this.#rank.get(entry.id) ?? UNRANKED;
			// Stable, so equally wanted chunks load in the order asked for.
			this.#queue.sort((a, b) => rank(a) - rank(b));
			this.#sorted = true;
		}
		return this.#queue.shift();
	}

	#pump(): void {
		while (this.#running < this.concurrency) {
			const entry = this.#next();
			if (!entry) return;
			this.#running += 1;
			let loading: Promise<Chunk>;
			try {
				loading = this.load(entry.id, entry.controller.signal);
			} catch (error) {
				// A loader that throws counts as a failed load.
				loading = Promise.reject(error);
			}
			loading
				.then((chunk) => {
					if (this.#pending.get(entry.id) !== entry) return;
					this.#pending.delete(entry.id);
					this.onLoad?.(entry.id, chunk);
					this.#drop(entry.id);
					this.#cache.set(entry.id, chunk);
					this.#bytes += chunk.data.byteLength;
					this.#evict();
					entry.resolve(chunk);
				})
				.catch((error: unknown) => {
					if (this.#pending.get(entry.id) === entry) this.#pending.delete(entry.id);
					entry.reject(error);
				})
				.finally(() => {
					this.#running -= 1;
					this.#pump();
				});
		}
	}

	/**
	 * Drop least recently used chunks until under budget, but never pinned
	 * chunks or chunks a view needs now (the budget stretches instead).
	 */
	#evict(): void {
		for (const [id, chunk] of this.#cache) {
			if (this.#bytes <= this.maxBytes) return;
			if (this.#pinned.has(id) || this.#isShown(id)) continue;
			this.#cache.delete(id);
			this.#bytes -= chunk.data.byteLength;
		}
	}
}
