/**
 * Sends label edits, undos, and redos to the server one at a time, in the
 * order they were made. Each carries a `client_op_id`, so a retry after a
 * lost answer applies once. Network failures and server errors retry with
 * backoff; edits not yet sent survive a reload (IndexedDB) and go out when
 * the page opens again.
 */

import { ApiError, api } from "#lib/api.ts";
import type { Vec3 } from "../viewer/tiles";
import type { DeltaIn } from "./deltas";

export const MAX_DELTAS = 512;

export interface OpOut {
	seq: number;
	chunks: { key: Vec3; version: number; sha: string | null }[];
}

interface QueuedEdit {
	kind: "edit";
	local: string;
	clientOpId: string;
	deltas: DeltaIn[];
	strict: boolean;
	tool: Record<string, unknown>;
}

interface QueuedToggle {
	kind: "undo" | "redo";
	local: string;
	clientOpId: string;
	/** The edit to undo or redo, by local id. */
	target: string;
}

export type Queued = QueuedEdit | QueuedToggle;

export type Outcome =
	| { op: Queued; result: OpOut }
	| { op: Queued; error: string; conflict?: Vec3[] };

/** Where unsent edits wait across reloads. */
export interface OpStorage {
	load(): Promise<QueuedEdit[]>;
	save(op: QueuedEdit): Promise<void>;
	remove(local: string): Promise<void>;
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

	#queue: Queued[] = [];
	#seqs = new Map<string, number>();
	#undo: string[][] = [];
	#redo: string[][] = [];
	#sending = false;
	#retryDelay = 0;
	#timer: ReturnType<typeof setTimeout> | null = null;
	#listeners = new Set<(outcome: Outcome) => void>();
	#stopped = false;

	constructor(
		private projectId: string,
		private storage: OpStorage | null = null,
		private send: Send = defaultSend,
	) {}

	/** Resume edits a previous page left unsent. */
	async start(): Promise<Queued[]> {
		const saved = (await this.storage?.load().catch(() => [])) ?? [];
		this.#queue.push(...saved);
		this.#changed();
		return saved;
	}

	stop(): void {
		this.#stopped = true;
		if (this.#timer) clearTimeout(this.#timer);
		this.#listeners.clear();
	}

	onOutcome(listener: (outcome: Outcome) => void): () => void {
		this.#listeners.add(listener);
		return () => this.#listeners.delete(listener);
	}

	/**
	 * Queue an edit; returns its ops' local ids (more than one when it
	 * touches more chunks than one op may carry; undo treats them as one).
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
		for (const op of ops) {
			this.#queue.push(op);
			this.storage?.save(op).catch(() => {});
		}
		this.#undo.push(ops.map((op) => op.local));
		this.#redo = [];
		this.#changed();
		return ops;
	}

	undo(): boolean {
		return this.#toggle(this.#undo, this.#redo, "undo");
	}

	redo(): boolean {
		return this.#toggle(this.#redo, this.#undo, "redo");
	}

	#toggle(from: string[][], to: string[][], kind: "undo" | "redo"): boolean {
		const group = from.pop();
		if (!group) return false;
		to.push(group);
		// Undo a multi-op edit last part first.
		const order = kind === "undo" ? [...group].reverse() : group;
		for (const target of order) this.#queue.push({ kind, local: newId(), clientOpId: newId(), target });
		this.#changed();
		return true;
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
				const outcome = await this.#sendOne(op);
				if (outcome === "retry") {
					this.#retryDelay = Math.min(30_000, Math.max(1000, this.#retryDelay * 2));
					this.#sending = false;
					this.#schedule(this.#retryDelay);
					return;
				}
				this.#retryDelay = 0;
				this.offline = false;
				this.#queue.shift();
				if (op.kind === "edit") this.storage?.remove(op.local).catch(() => {});
				this.pending = this.#queue.length;
				for (const listener of this.#listeners) listener(outcome);
			}
		} finally {
			this.#sending = false;
		}
	}

	async #sendOne(op: Queued): Promise<Outcome | "retry"> {
		const base = `/api/projects/${this.projectId}/labels`;
		try {
			let result: OpOut;
			if (op.kind === "edit") {
				result = await this.send(`${base}/ops`, {
					client_op_id: op.clientOpId,
					deltas: op.deltas,
					strict: op.strict,
					tool: op.tool,
				});
				this.#seqs.set(op.local, result.seq);
			} else {
				const seq = this.#seqs.get(op.target);
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
			return { op, error };
		}
	}
}

/** Unsent edits in IndexedDB, per project; null where IndexedDB doesn't work. */
export function indexedDbStorage(projectId: string): OpStorage | null {
	if (typeof indexedDB === "undefined") return null;
	const open = () =>
		new Promise<IDBDatabase>((resolve, reject) => {
			const request = indexedDB.open("m4p-label-ops", 1);
			request.onupgradeneeded = () => {
				const store = request.result.createObjectStore("ops", { keyPath: "local" });
				store.createIndex("project", "project");
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
	return {
		async load() {
			const rows = await run("readonly", (store) => store.index("project").getAll(projectId));
			return (rows as (QueuedEdit & { project: string; at: number })[])
				.sort((a, b) => a.at - b.at)
				.map(({ project: _project, at: _at, ...op }) => op);
		},
		async save(op) {
			await run("readwrite", (store) => store.put({ ...op, project: projectId, at: Date.now() + Math.random() / 1000 }));
		},
		async remove(local) {
			await run("readwrite", (store) => store.delete(local));
		},
	};
}
