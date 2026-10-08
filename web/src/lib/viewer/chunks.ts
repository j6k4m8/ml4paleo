/**
 * A cache of decoded chunks, kept under a byte budget (least recently used
 * chunks go first), that loads each chunk once however often it is asked
 * for, runs a limited number of loads at a time, most wanted first, drops
 * loads that no view needs any more, and asks again, without holding one of
 * its places meanwhile, for chunks a server is still making (`Busy`).
 */

export interface Chunk {
	data: ArrayBufferView & { length: number };
	/** Shape of the chunk, (z, y, x). Edge chunks may be smaller. */
	shape: number[];
	/** For label chunks, the version the server served. */
	version?: number;
	/**
	 * For a coarser level of the labels, which the server makes from the
	 * chunks under it, what it was made from (its `X-Pyramid-Version`). That
	 * isn't the chunk's `version`, which an edit's base version comes from.
	 */
	pyramid?: number;
}

export type Loader = (id: string, signal: AbortSignal) => Promise<Chunk>;

/**
 * What a loader throws for a chunk that isn't ready but will be (a coarser
 * level of the labels, which the server makes from the chunks under it, and
 * answers 503 while it works): the store asks again after `delay`
 * milliseconds, and the chunk takes no place in line meanwhile. `cause` is
 * what the server answered. A chunk still busy after `PATIENCE_MS` fails
 * with the last `Busy`.
 */
export class Busy extends Error {
	constructor(
		readonly delay: number,
		cause?: unknown,
	) {
		super("The server is still making this chunk", { cause });
		this.name = "Busy";
	}
}

/**
 * A limit on the loads of chunks a server takes long to make, so that
 * whatever it only has to read never waits for them: while a chunk that isn't
 * slow is queued, or shown (see `want`) and not loaded yet, the slow ones
 * take no more than `places` of the store's places, which leaves the rest for
 * it. With none waiting they may take them all.
 */
export interface SlowLimit {
	/** Whether the server takes long to make a chunk. */
	slow: (id: string) => boolean;
	/** The most places the loads of such chunks take at once while others wait. */
	places: number;
}

/** The longest a chunk answered busy is waited for, counted from the start of its first load. */
export const PATIENCE_MS = 120_000;
/**
 * The least time between one chunk being asked for again and the next, across
 * the store: however many wait, a busy server sees only a trickle of asks.
 */
export const RETRY_GAP_MS = 250;

interface Pending {
	id: string;
	promise: Promise<Chunk>;
	resolve: (chunk: Chunk) => void;
	reject: (error: unknown) => void;
	controller: AbortController;
	/** Replaces a cached copy that is now out of date. */
	refresh: boolean;
	/** For a refresh, the version the new copy is asked for (see `refresh`). */
	version?: number;
	/** Whether the server takes long to make it (see `SlowLimit`). */
	slow: boolean;
	/** When its first load began, which its patience is counted from. */
	began?: number;
	/**
	 * While a load answered busy waits to be asked again (in the queue, with
	 * no place taken): the earliest it may be.
	 */
	retryAt?: number;
}

// Where a chunk nobody wants waits: after every wanted one.
const UNRANKED = Number.MAX_SAFE_INTEGER;

export class ChunkStore {
	#cache = new Map<string, Chunk>();
	#bytes = 0;
	#pending = new Map<string, Pending>();
	// Loads waiting for a place, and loads waiting to be asked again.
	#queue: Pending[] = [];
	#sorted = true;
	#running = 0;
	// How many of the loads running are of slow chunks (see `SlowLimit`).
	#runningSlow = 0;
	// When the next load asked again may start, and the timer that wakes the
	// queue for one that isn't due yet (with the time it is set for).
	#retryFrom = 0;
	#timer: ReturnType<typeof setTimeout> | undefined;
	#wakeAt = Number.POSITIVE_INFINITY;
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
		private limit?: SlowLimit,
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

	/** The ids of the cached chunks, the least recently used first. */
	ids(): IterableIterator<string> {
		return this.#cache.keys();
	}

	/** Whether a load of the chunk is queued, running, or waiting to be asked again. */
	isLoading(id: string): boolean {
		return this.#pending.has(id);
	}

	/** The load of a chunk queued, running or waiting to be asked again, if there is one (without starting one). */
	loading(id: string): Promise<Chunk> | undefined {
		return this.#pending.get(id)?.promise;
	}

	/** The version a queued or running refresh of a chunk was asked for, if any. */
	refreshing(id: string): number | undefined {
		const entry = this.#pending.get(id);
		return entry?.refresh ? entry.version : undefined;
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
	 * cached copy in use until the new one replaces it. `version` notes what
	 * the new copy will be at least (a label chunk's version, say), for
	 * `refreshing`.
	 */
	refresh(id: string, version?: number): Promise<Chunk> {
		this.#cancel(id, "Changed while loading", false);
		return this.#enqueue(id, true, version);
	}

	/** Whether any view needs this chunk now. */
	isWanted(id: string): boolean {
		return this.#rank.has(id);
	}

	#enqueue(id: string, refresh: boolean, version?: number): Promise<Chunk> {
		let resolve!: (chunk: Chunk) => void;
		let reject!: (error: unknown) => void;
		const promise = new Promise<Chunk>((ok, fail) => {
			resolve = ok;
			reject = fail;
		});
		const entry: Pending = { id, promise, resolve, reject, controller: new AbortController(), refresh, version, slow: this.limit?.slow(id) ?? false };
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
		// A view that no longer shows a chunk it waits for frees the places held for it.
		this.#pump();
	}

	#isShown(id: string): boolean {
		for (const set of this.#protected.values()) if (set.has(id)) return true;
		return false;
	}

	/** Cancel queued, running and waiting loads of chunks not in `wanted`. */
	keepOnly(wanted: Set<string>): void {
		const cancelled = [...this.#pending.values()].filter((entry) => !wanted.has(entry.id));
		if (cancelled.length === 0) return;
		const gone = new Set(cancelled);
		this.#queue = this.#queue.filter((queued) => !gone.has(queued));
		for (const entry of cancelled) this.#stop(entry, "No longer needed", true);
		this.#unwake();
	}

	#cancel(id: string, reason: string, dropStale = true): void {
		const entry = this.#pending.get(id);
		if (!entry) return;
		this.#queue = this.#queue.filter((queued) => queued !== entry);
		this.#stop(entry, reason, dropStale);
		this.#unwake();
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

	/**
	 * The queued load to start next: the most wanted, and any never started
	 * before those asked again (in the order they came due), which wait
	 * until they are due, a few at a time. A slow chunk's load that would take
	 * more of the places than it may, with a chunk that isn't slow waiting for
	 * one, is passed over.
	 */
	#next(): Pending | undefined {
		if (!this.#sorted) {
			const rank = (entry: Pending) => this.#rank.get(entry.id) ?? UNRANKED;
			// Stable, so equally wanted chunks load in the order asked for.
			this.#queue.sort((a, b) => {
				if (a.retryAt === undefined) return b.retryAt === undefined ? rank(a) - rank(b) : -1;
				return b.retryAt === undefined ? 1 : a.retryAt - b.retryAt;
			});
			this.#sorted = true;
		}
		const now = Date.now();
		// Whether places are held for chunks that aren't slow, looked into only if it matters.
		let held: boolean | undefined;
		for (const [index, entry] of this.#queue.entries()) {
			if (entry.slow && this.limit && this.#runningSlow >= this.limit.places) {
				held ??= this.#wantsFast();
				if (held) continue;
			}
			if (entry.retryAt !== undefined) {
				const at = Math.max(entry.retryAt, this.#retryFrom);
				// The ones after it come due later still.
				if (at > now) return this.#wake(at - now);
				this.#retryFrom = now + RETRY_GAP_MS;
			}
			this.#queue.splice(index, 1);
			return entry;
		}
		return undefined;
	}

	/**
	 * Whether a chunk that isn't slow is waiting for a place: queued, or shown
	 * by a view (which asks for it next) and neither loaded nor loading.
	 */
	#wantsFast(): boolean {
		if (this.#queue.some((entry) => !entry.slow)) return true;
		for (const shown of this.#protected.values()) {
			for (const id of shown) {
				if (!this.#cache.has(id) && !this.#pending.has(id) && !this.limit?.slow(id)) return true;
			}
		}
		return false;
	}

	/** Start the queue again in `ms`, unless it is to be woken sooner anyway. */
	#wake(ms: number): undefined {
		const at = Date.now() + ms;
		if (at >= this.#wakeAt) return;
		clearTimeout(this.#timer);
		this.#wakeAt = at;
		this.#timer = setTimeout(() => {
			this.#timer = undefined;
			this.#wakeAt = Number.POSITIVE_INFINITY;
			this.#pump();
		}, ms);
	}

	/** Stop the timer once nothing is queued for it to wake, so a store nobody uses isn't kept alive by it. */
	#unwake(): void {
		if (this.#queue.length > 0) return;
		clearTimeout(this.#timer);
		this.#timer = undefined;
		this.#wakeAt = Number.POSITIVE_INFINITY;
	}

	/**
	 * Queue a load that was answered busy to be asked again, unless the wait
	 * would end past its patience.
	 */
	#park(entry: Pending, busy: Busy): boolean {
		const now = Date.now();
		if (now + busy.delay - (entry.began ?? now) > PATIENCE_MS) return false;
		entry.retryAt = now + busy.delay;
		this.#queue.push(entry);
		this.#sorted = false;
		return true;
	}

	#pump(): void {
		while (this.#running < this.concurrency) {
			const entry = this.#next();
			if (!entry) break;
			entry.began ??= Date.now();
			entry.retryAt = undefined;
			this.#running += 1;
			if (entry.slow) this.#runningSlow += 1;
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
					if (this.#pending.get(entry.id) === entry) {
						// The server is making it: wait to be asked again, in no one's way.
						if (error instanceof Busy && this.#park(entry, error)) return;
						this.#pending.delete(entry.id);
					}
					entry.reject(error);
				})
				.finally(() => {
					this.#running -= 1;
					if (entry.slow) this.#runningSlow -= 1;
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
