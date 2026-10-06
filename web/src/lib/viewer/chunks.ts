/**
 * A cache of decoded chunks, kept under a byte budget (least recently used
 * chunks go first), that loads each chunk once however often it is asked
 * for, runs a limited number of loads at a time, and drops loads that the
 * view no longer needs.
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
	resolve: (chunk: Chunk) => void;
	reject: (error: unknown) => void;
	controller: AbortController;
	started: boolean;
	/** Replaces a cached copy that is now out of date. */
	refresh: boolean;
}

export class ChunkStore {
	#cache = new Map<string, Chunk>();
	#bytes = 0;
	#pending = new Map<string, { promise: Promise<Chunk>; entry: Pending }>();
	#queue: Pending[] = [];
	#running = 0;
	#pinned = new Set<string>();
	#wanted = new Map<string, Set<string>>();
	#protected = new Map<string, Set<string>>();
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

	/** Load a chunk (once), queueing behind loads asked for earlier. */
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
		const pending = this.#pending.get(id);
		if (pending) {
			pending.entry.controller.abort();
			this.#pending.delete(id);
			this.#queue = this.#queue.filter((entry) => entry !== pending.entry);
			pending.entry.reject(new DOMException("Changed while loading", "AbortError"));
		}
		return this.#enqueue(id, true);
	}

	/** Whether any view needs this chunk now. */
	isWanted(id: string): boolean {
		for (const set of this.#wanted.values()) if (set.has(id)) return true;
		return false;
	}

	#enqueue(id: string, refresh: boolean): Promise<Chunk> {
		let entry!: Pending;
		const promise = new Promise<Chunk>((resolve, reject) => {
			entry = { id, resolve, reject, controller: new AbortController(), started: false, refresh };
		});
		this.#pending.set(id, { promise, entry });
		this.#queue.push(entry);
		this.#pump();
		return promise;
	}

	/**
	 * Say which chunks `owner` (for example one of several views sharing
	 * this store) needs now; loads no owner needs are cancelled. Of those,
	 * the `shown` ones (by default all) are never evicted, even over budget.
	 */
	want(owner: string, ids: Set<string>, shown: Set<string> = ids): void {
		this.#wanted.set(owner, ids);
		this.#protected.set(owner, shown);
		const union = new Set<string>();
		for (const set of this.#wanted.values()) for (const id of set) union.add(id);
		this.keepOnly(union);
	}

	#isShown(id: string): boolean {
		for (const set of this.#protected.values()) if (set.has(id)) return true;
		return false;
	}

	/** Cancel queued and running loads of chunks not in `wanted`. */
	keepOnly(wanted: Set<string>): void {
		for (const [id, { entry }] of this.#pending) {
			if (wanted.has(id)) continue;
			entry.controller.abort();
			this.#pending.delete(id);
			// A cancelled refresh leaves an out-of-date copy: drop it, so the
			// next request loads afresh.
			if (entry.refresh) this.#drop(id);
			entry.reject(new DOMException("No longer needed", "AbortError"));
		}
		this.#queue = this.#queue.filter((entry) => wanted.has(entry.id));
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
		const pending = this.#pending.get(id);
		if (pending) {
			pending.entry.controller.abort();
			this.#pending.delete(id);
			this.#queue = this.#queue.filter((entry) => entry !== pending.entry);
			pending.entry.reject(new DOMException("Changed while loading", "AbortError"));
		}
	}

	#drop(id: string): void {
		const chunk = this.#cache.get(id);
		if (chunk) {
			this.#cache.delete(id);
			this.#bytes -= chunk.data.byteLength;
		}
	}

	#pump(): void {
		while (this.#running < this.concurrency) {
			const entry = this.#queue.shift();
			if (!entry) return;
			if (entry.controller.signal.aborted) continue;
			entry.started = true;
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
					if (this.#pending.get(entry.id)?.entry !== entry) return;
					this.#pending.delete(entry.id);
					this.onLoad?.(entry.id, chunk);
					const old = this.#cache.get(entry.id);
					if (old) {
						this.#cache.delete(entry.id);
						this.#bytes -= old.data.byteLength;
					}
					this.#cache.set(entry.id, chunk);
					this.#bytes += chunk.data.byteLength;
					this.#evict();
					entry.resolve(chunk);
				})
				.catch((error: unknown) => {
					if (this.#pending.get(entry.id)?.entry === entry) this.#pending.delete(entry.id);
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
