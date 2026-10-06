/**
 * Sends label edits, undos, and redos to the server one at a time, in the
 * order they were made. Each carries a `client_op_id`, so a retry after a
 * lost answer applies once. Network failures and server errors retry with
 * backoff; ops not yet sent survive a reload (IndexedDB, per user and
 * project) and go out when the page opens again.
 *
 * Undoing an edit that hasn't been sent yet just drops it; redoing it sends
 * it as a new op.
 */

import { ApiError, api } from "#lib/api.ts";
import type { Vec3 } from "../viewer/tiles";
import type { DeltaIn } from "./deltas";

export const MAX_DELTAS = 512;

export interface OpOut {
	seq: number;
	chunks: { key: Vec3; version: number; sha: string | null }[];
}

export interface QueuedEdit {
	kind: "edit";
	local: string;
	clientOpId: string;
	deltas: DeltaIn[];
	strict: boolean;
	tool: Record<string, unknown>;
}

export interface QueuedToggle {
	kind: "undo" | "redo";
	local: string;
	clientOpId: string;
	/** The edit to undo or redo, by local id. */
	target: string;
	/** Its seq, once known (so the toggle can be saved across reloads). */
	seq?: number;
}

export type Queued = QueuedEdit | QueuedToggle;

export type Outcome =
	| { op: Queued; result: OpOut }
	| { op: Queued; error: string; conflict?: Vec3[]; alreadyDone?: boolean }
	| { op: QueuedEdit; cancelled: true };

/** Where unsent ops wait across reloads. */
export interface OpStorage {
	load(): Promise<Queued[]>;
	save(op: Queued): Promise<void>;
	remove(local: string): Promise<void>;
}

interface UndoGroup {
	ops: QueuedEdit[];
	/** The edits were undone before they were sent, so they were dropped. */
	dropped?: boolean;
}

type Send = (path: string, body: unknown) => Promise<OpOut>;

const defaultSend: Send = (path, body) => api<OpOut>(path, { body });

function newId(): string {
	return crypto.randomUUID();
}

export class OpQueue {
	/** Ops not yet confirmed by the server. */
	pending = $state(0);
	offline = $state(false);
	error = $state("");
	/** How many edits can be undone and redone. */
	undoable = $state(0);
	redoable = $state(0);
	/** Last chance to adjust an edit (for example its base versions) before it goes. */
	beforeSend: ((op: QueuedEdit) => QueuedEdit) | null = null;

	#queue: Queued[] = [];
	#inFlight: Queued | null = null;
	#seqs = new Map<string, number>();
	#undo: UndoGroup[] = [];
	#redo: UndoGroup[] = [];
	#sending = false;
	#retryDelay = 0;
	#timer: ReturnType<typeof setTimeout> | null = null;
	#listeners = new Set<(outcome: Outcome) => void>();
	#requeueListeners = new Set<(ops: QueuedEdit[]) => void>();
	#stopped = false;

	constructor(
		private projectId: string,
		private storage: OpStorage | null = null,
		private send: Send = defaultSend,
	) {}

	/** Resume ops a previous page left unsent, ahead of anything new. */
	async start(): Promise<Queued[]> {
		const saved = (await this.storage?.load().catch(() => [])) ?? [];
		this.#queue.unshift(...saved);
		this.#changed();
		return saved;
	}

	stop(): void {
		this.#stopped = true;
		if (this.#timer) clearTimeout(this.#timer);
		this.#listeners.clear();
		this.#requeueListeners.clear();
	}

	onOutcome(listener: (outcome: Outcome) => void): () => void {
		this.#listeners.add(listener);
		return () => this.#listeners.delete(listener);
	}

	/** Edits queued again by a redo of edits that were dropped unsent. */
	onRequeue(listener: (ops: QueuedEdit[]) => void): () => void {
		this.#requeueListeners.add(listener);
		return () => this.#requeueListeners.delete(listener);
	}

	/**
	 * Queue an edit; returns its ops (more than one when it touches more
	 * chunks than one op may carry; undo treats them as one).
	 */
	edit(deltas: DeltaIn[], options: { strict?: boolean; tool?: Record<string, unknown> } = {}): QueuedEdit[] {
		const ops: QueuedEdit[] = [];
		for (let i = 0; i < deltas.length; i += MAX_DELTAS) {
			ops.push({
				kind: "edit",
				local: newId(),
				clientOpId: newId(),
				deltas: deltas.slice(i, i + MAX_DELTAS),
				strict: options.strict ?? false,
				tool: options.tool ?? {},
			});
		}
		if (ops.length === 0) return ops;
		this.#enqueue(ops);
		this.#undo.push({ ops });
		this.#redo = [];
		this.#changed();
		return ops;
	}

	undo(): boolean {
		const group = this.#undo.pop();
		if (!group) return false;
		const waiting = group.ops.every((op) => this.#queue.includes(op) && op !== this.#inFlight);
		if (waiting) {
			// Never sent: drop the edits instead of undoing them.
			this.#queue = this.#queue.filter((op) => !group.ops.includes(op as QueuedEdit));
			for (const op of group.ops) {
				this.storage?.remove(op.local).catch(() => {});
				this.#emit({ op, cancelled: true });
			}
			this.#redo.push({ ops: group.ops, dropped: true });
		} else {
			this.#toggle(group, "undo");
			this.#redo.push(group);
		}
		this.#changed();
		return true;
	}

	redo(): boolean {
		const group = this.#redo.pop();
		if (!group) return false;
		if (group.dropped) {
			const ops = group.ops.map((op) => ({ ...op, local: newId(), clientOpId: newId() }));
			this.#enqueue(ops);
			for (const listener of this.#requeueListeners) listener(ops);
			this.#undo.push({ ops });
		} else {
			this.#toggle(group, "redo");
			this.#undo.push(group);
		}
		this.#changed();
		return true;
	}

	#enqueue(ops: Queued[]): void {
		for (const op of ops) {
			this.#queue.push(op);
			this.storage?.save(op).catch(() => {});
		}
	}

	#toggle(group: UndoGroup, kind: "undo" | "redo"): void {
		// Undo a multi-op edit last part first.
		const order = kind === "undo" ? [...group.ops].reverse() : group.ops;
		for (const target of order) {
			const seq = this.#seqs.get(target.local);
			const op: QueuedToggle = { kind, local: newId(), clientOpId: newId(), target: target.local, seq };
			this.#queue.push(op);
			// Without a seq a reloaded page couldn't tell which edit it means.
			if (seq !== undefined) this.storage?.save(op).catch(() => {});
		}
	}

	#emit(outcome: Outcome): void {
		for (const listener of this.#listeners) listener(outcome);
	}

	#changed(): void {
		this.pending = this.#queue.length;
		this.undoable = this.#undo.length;
		this.redoable = this.#redo.length;
		this.#schedule(0);
	}

	#schedule(delay: number): void {
		if (this.#stopped || this.#sending || this.#timer) return;
		this.#timer = setTimeout(() => {
			this.#timer = null;
			void this.#flush();
		}, delay);
	}

	async #flush(): Promise<void> {
		if (this.#sending || this.#stopped) return;
		this.#sending = true;
		try {
			while (this.#queue.length > 0 && !this.#stopped) {
				const op = this.#queue[0]!;
				this.#inFlight = op;
				const outcome = await this.#sendOne(op);
				this.#inFlight = null;
				if (outcome === "retry") {
					this.#retryDelay = Math.min(30_000, Math.max(1000, this.#retryDelay * 2));
					this.#sending = false;
					this.#schedule(this.#retryDelay);
					return;
				}
				this.#retryDelay = 0;
				this.offline = false;
				this.#queue.shift();
				this.storage?.remove(op.local).catch(() => {});
				this.pending = this.#queue.length;
				this.#emit(outcome);
			}
		} finally {
			this.#inFlight = null;
			this.#sending = false;
		}
	}

	async #sendOne(op: Queued): Promise<Outcome | "retry"> {
		const base = `/api/projects/${this.projectId}/labels`;
		try {
			let result: OpOut;
			if (op.kind === "edit") {
				const ready = this.beforeSend ? this.beforeSend(op) : op;
				result = await this.send(`${base}/ops`, {
					client_op_id: ready.clientOpId,
					deltas: ready.deltas,
					strict: ready.strict,
					tool: ready.tool,
				});
				this.#seqs.set(op.local, result.seq);
			} else {
				const seq = op.seq ?? this.#seqs.get(op.target);
				// The edit never applied, so there is nothing to undo or redo.
				if (seq === undefined) return { op, error: "That edit didn't save." };
				result = await this.send(`${base}/ops/${seq}/${op.kind}`, { client_op_id: op.clientOpId });
			}
			this.error = "";
			return { op, result };
		} catch (e) {
			if (!(e instanceof ApiError)) {
				// fetch failed: offline, or the server is unreachable.
				this.offline = true;
				return "retry";
			}
			if (e.status >= 500 || e.status === 429) return "retry";
			if (e.status === 401 || e.status === 403) {
				this.error = "Sign in again to save your labels.";
				return "retry";
			}
			const detail = e.detail as { message?: string; chunks?: Vec3[] } | string;
			if (typeof detail === "object" && detail?.chunks) {
				return { op, error: detail.message ?? "Some chunks changed.", conflict: detail.chunks };
			}
			const error = typeof detail === "string" ? detail : `HTTP ${e.status}`;
			if (op.kind === "edit") this.error = error;
			return { op, error, alreadyDone: op.kind !== "edit" && e.status === 409 };
		}
	}
}

const DATABASE = "m4p-label-ops";

/**
 * Unsent ops in IndexedDB, for one user in one project; null where
 * IndexedDB doesn't work.
 */
export function indexedDbStorage(userId: string, projectId: string): OpStorage | null {
	if (typeof indexedDB === "undefined") return null;
	const owner = `${userId}/${projectId}`;
	const open = () =>
		new Promise<IDBDatabase>((resolve, reject) => {
			const request = indexedDB.open(DATABASE, 2);
			request.onupgradeneeded = () => {
				const db = request.result;
				if (db.objectStoreNames.contains("ops")) db.deleteObjectStore("ops");
				db.createObjectStore("ops", { keyPath: "local" }).createIndex("owner", "owner");
			};
			request.onsuccess = () => resolve(request.result);
			request.onerror = () => reject(request.error);
		});
	const run = async <T>(mode: IDBTransactionMode, act: (store: IDBObjectStore) => IDBRequest<T>) => {
		const db = await open();
		return new Promise<T>((resolve, reject) => {
			const request = act(db.transaction("ops", mode).objectStore("ops"));
			request.onsuccess = () => resolve(request.result);
			request.onerror = () => reject(request.error);
		}).finally(() => db.close());
	};
	let counter = 0;
	return {
		async load() {
			const rows = await run("readonly", (store) => store.index("owner").getAll(owner));
			return (rows as (Queued & { owner: string; at: number })[])
				.sort((a, b) => a.at - b.at)
				.map(({ owner: _owner, at: _at, ...op }) => op as Queued);
		},
		async save(op) {
			await run("readwrite", (store) => store.put({ ...op, owner, at: Date.now() * 1000 + (counter++ % 1000) }));
		},
		async remove(local) {
			await run("readwrite", (store) => store.delete(local));
		},
	};
}

/** Forget every unsent op in this browser (on sign-out). */
export function clearOpStorage(): void {
	try {
		indexedDB?.deleteDatabase(DATABASE);
	} catch {
		// Nothing to clear.
	}
}
