/** Explicit training follows the signed-in user across routes, not the Models page's lifetime. */
import { ApiError, api } from "./api";

export interface TrainingModel {
	id: string;
	name: string;
	status: "training" | "ready" | "failed";
	error?: string | null;
	live?: boolean;
	created_by?: string | null;
}

interface PendingTraining {
	project: string;
	model: string;
	name: string;
}

export interface TrainingNotice extends PendingTraining {
	id: string;
	kind: "started" | "ready" | "failed" | "unavailable";
	message: string;
}

type Storage = Pick<globalThis.Storage, "getItem" | "setItem">;
const POLL_MS = 3000;
const keyOf = (job: PendingTraining) => `${job.project}/${job.model}`;
const safeId = (value: unknown): value is string => typeof value === "string" && /^[a-zA-Z0-9_-]+$/.test(value);

export class TrainingNotifications {
	pending: PendingTraining[] = $state([]);
	submitting: string[] = $state([]);
	notices: TrainingNotice[] = $state([]);
	#user: string | null = null;
	#storage: Storage | null = null;
	#controller = new AbortController();
	#generation = 0;
	#timer: ReturnType<typeof setTimeout> | undefined;
	#polling = false;
	#finished = new Set<string>();
	#dismissTimers = new Map<string, ReturnType<typeof setTimeout>>();

	start(user: string, storage?: Storage | null): void {
		if (user === this.#user) return;
		this.stop();
		this.#user = user;
		this.#controller = new AbortController();
		try { this.#storage = storage === undefined ? globalThis.sessionStorage : storage; } catch { this.#storage = null; }
		try {
			const saved: unknown = JSON.parse(this.#storage?.getItem(this.storageKey) ?? "[]");
			const record = saved && typeof saved === "object" ? saved as { pending?: unknown; notices?: unknown } : {};
			const pending = Array.isArray(saved) ? saved : record.pending;
			if (Array.isArray(pending)) {
				const unique = new Map<string, PendingTraining>();
				for (const job of pending.slice(0, 100)) {
					if (!job || !safeId(job.project) || !safeId(job.model) || typeof job.name !== "string") continue;
					unique.set(keyOf(job), { project: job.project, model: job.model, name: job.name.slice(0, 100) });
				}
				this.pending = [...unique.values()];
			}
			if (Array.isArray(record.notices)) {
				for (const notice of record.notices.slice(-100)) {
					if (!notice || !safeId(notice.project) || !safeId(notice.model) || typeof notice.name !== "string"
						|| !["ready", "failed", "unavailable"].includes(notice.kind) || typeof notice.message !== "string") continue;
					const id = keyOf(notice);
					this.#finished.add(id);
					this.notices = [...this.notices.filter((item) => item.id !== id), { project: notice.project, model: notice.model,
						name: notice.name.slice(0, 100), id, kind: notice.kind, message: notice.message.slice(0, 500) }];
				}
				this.pending = this.pending.filter((job) => !this.#finished.has(keyOf(job)));
			}
		} catch { /* Unavailable or old browser storage must not block training. */ }
		this.schedule(0);
	}

	/** Account changes abort reads and prevent late replies or notices crossing accounts. */
	stop(): void {
		this.#generation++;
		this.#controller.abort();
		clearTimeout(this.#timer);
		this.#timer = undefined;
		for (const timer of this.#dismissTimers.values()) clearTimeout(timer);
		this.#dismissTimers.clear();
		this.#finished.clear();
		this.#user = null;
		this.#storage = null;
		this.#polling = false;
		this.pending = [];
		this.submitting = [];
		this.notices = [];
	}

	private get storageKey(): string { return `ml4paleo:training:${this.#user}`; }
	private persist(): void {
		try { this.#storage?.setItem(this.storageKey, JSON.stringify({ pending: this.pending,
			notices: this.notices.filter((notice) => notice.kind !== "started").slice(-100) })); } catch { /* Best effort. */ }
	}

	isTraining(project: string): boolean {
		return this.submitting.includes(project) || this.pending.some((job) => job.project === project);
	}

	/** Register existing manual work when visiting Models; never notify for automatic Live cycles. */
	watch(project: string, model: TrainingModel): void {
		if (!this.#user || model.live || model.created_by !== this.#user || model.status !== "training") return;
		this.track({ project, model: model.id, name: model.name });
	}

	private track(job: PendingTraining): void {
		const key = keyOf(job);
		if (this.#finished.has(key) || this.pending.some((item) => keyOf(item) === key)) return;
		this.pending = [...this.pending, job];
		this.persist();
		this.schedule();
	}

	/** The mutex covers the POST and the whole training, even if the page unmounts. */
	async submit(project: string, body: { plugin: string; params: Record<string, number>; name?: string }): Promise<TrainingModel | null> {
		if (this.isTraining(project)) return null;
		if (!this.#user) throw new Error("Sign in before training a model.");
		const generation = this.#generation;
		this.submitting = [...this.submitting, project];
		try {
			const model = await api<TrainingModel>(`/api/projects/${project}/models`, { body, signal: this.#controller.signal });
			if (generation !== this.#generation) return null;
			const job = { project, model: model.id, name: model.name };
			if (model.status === "training") {
				this.track(job);
				this.notify(job, "started", "Model is training, you will get a notification when training is complete!");
			} else this.finish(job, model);
			return model;
		} catch (error) {
			if (generation !== this.#generation) return null;
			throw error;
		} finally {
			if (generation === this.#generation) this.submitting = this.submitting.filter((id) => id !== project);
		}
	}

	dismiss(id: string): void {
		clearTimeout(this.#dismissTimers.get(id));
		this.#dismissTimers.delete(id);
		this.notices = this.notices.filter((notice) => notice.id !== id);
		this.persist();
	}

	private notify(job: PendingTraining, kind: TrainingNotice["kind"], message: string): void {
		const id = keyOf(job);
		this.dismiss(id);
		this.notices = [...this.notices, { ...job, id, kind, message }];
		this.persist();
		// Completion/failure stay until dismissed, so returning to a background tab cannot miss them.
		if (kind === "started") this.#dismissTimers.set(id, setTimeout(() => this.dismiss(id), 8000));
	}

	private finish(job: PendingTraining, model: TrainingModel): void {
		if (this.#finished.has(keyOf(job))) return;
		this.#finished.add(keyOf(job));
		this.pending = this.pending.filter((item) => keyOf(item) !== keyOf(job));
		this.persist();
		this.notify({ ...job, name: model.name }, model.status === "ready" ? "ready" : "failed",
			model.status === "ready" ? `Training complete — ${model.name} is ready.`
				: model.error === "Stopped." ? `Training stopped for ${model.name}.`
					: `Training failed for ${model.name}. Open Models for details.`);
	}

	private schedule(delay = POLL_MS): void {
		if (!this.#user || !this.pending.length || this.#timer !== undefined || this.#polling) return;
		this.#timer = setTimeout(() => { this.#timer = undefined; void this.poll(); }, delay);
	}

	private async poll(): Promise<void> {
		const generation = this.#generation;
		this.#polling = true;
		let retry = false;
		await Promise.all(this.pending.map(async (job) => {
			try {
				const model = await api<TrainingModel>(`/api/projects/${job.project}/models/${job.model}`, { signal: this.#controller.signal });
				if (generation !== this.#generation) return;
				if (model.status !== "training") this.finish(job, model);
			} catch (error) {
				if (generation !== this.#generation) return;
				if (error instanceof ApiError && [403, 404].includes(error.status)) {
					this.#finished.add(keyOf(job));
					this.pending = this.pending.filter((item) => keyOf(item) !== keyOf(job));
					this.persist();
					this.notify(job, "unavailable", `Cannot check ${job.name}. It may have been removed, or your access changed.`);
				} else retry = true; // Keep tracking through connection/server/session failures.
			}
		}));
		if (generation !== this.#generation) return;
		this.#polling = false;
		this.schedule(retry ? 10000 : POLL_MS);
	}
}

export const trainingNotifications = new TrainingNotifications();
