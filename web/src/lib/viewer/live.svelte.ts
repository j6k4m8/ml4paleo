import { api, message } from "#lib/api.ts";
import type { Pipeline } from "#lib/types.ts";
import type { Box } from "../rois.svelte";
import { ChunkStore } from "./chunks";
import { absolute } from "./image";
import { labelLoader, type WorkerPool } from "./loader";
import type { LiveModel, ModelPlugin } from "./live";
import type { Vec3 } from "./tiles";

export interface LiveRegion {
	artifact_id: string;
	zarr_url: string;
	box: Box;
	store: ChunkStore;
	model_id: string;
	model_name: string;
	label_seq: number;
}
interface ChunkResult extends Pick<LiveRegion, "artifact_id" | "zarr_url" | "box"> { ready: boolean; pipeline_id: string }
export type LearningPhase = "loading" | "learning" | "queued" | "ready" | "manual" | "error" | "waiting";
export type PredictionPhase = "predicting" | "queued" | "ready" | "waiting" | "retrying";

/** Separate learning and inference loops. Models and chunks are immutable;
 * publication is guarded by selection generation and acceptance's freeze.
 * The browser coalesces intent; the server also enforces limits across tabs.
 */
export class LivePreview {
	enabled = $state(false);
	paused = $state(false);
	plugin = $state<ModelPlugin | null>(null);
	model = $state.raw<LiveModel | null>(null);
	learning = $state("Waiting to start.");
	predicting = $state("Waiting to start.");
	learningError = $state("");
	predictionError = $state("");
	get error() { return [this.learningError, this.predictionError].filter(Boolean).join(" "); }
	store = $state.raw<ChunkStore | null>(null);
	regions = $state.raw<ReadonlyMap<string, LiveRegion>>(new Map());
	wanted = $state<string[]>([]);
	#revision = $state(0);
	#loadingRegion: ChunkStore | null = null;
	/** The artifact actually displayed in each chunk, including older checkpoints. */
	get versions(): ReadonlyMap<string, string> { return new Map([...this.regions].map(([id, region]) => [id, region.artifact_id])); }
	get stale(): ReadonlySet<string> {
		return new Set([...this.regions].filter(([, region]) => region.model_id !== this.model?.id || region.label_seq < this.#revision).map(([id]) => id));
	}
	/** Don't imply progress while learning is blocked or this backend needs manual training. */
	get updating(): boolean {
		if (!this.enabled || this.paused || this.predictionError || this.#freeze()) return false;
		return [...this.regions.values()].some((region) => region.model_id !== this.model?.id)
			|| (!this.learningError && this.plugin?.capabilities.learning !== "manual" && this.stale.size > 0);
	}
	#coveredRevision = -1;
	#dirty = $state(true);
	#changedAt = 0;
	#lastTrain = 0;
	#generation = 0;
	#learningBusy = $state(false);
	#predictionBusy = $state(false);
	#selecting = $state(false);
	#stopped = false;
	#training = $state.raw<LiveModel | null>(null);
	#controller = new AbortController();
	#timer: ReturnType<typeof setInterval>;
	#learningRetryAt = 0;
	#predictionRetryAt = 0;
	#freeze: () => boolean;

	/** Compact presentation follows work state, never parses the explanatory copy. */
	get learningPhase(): LearningPhase {
		if (this.learningError) return "error";
		if (this.#selecting) return "loading";
		if (this.#training?.status === "training" || this.#learningBusy) return "learning";
		if (this.plugin?.capabilities.learning === "manual") return this.model ? "ready" : "manual";
		if (this.#dirty && this.plugin) return "queued";
		return this.model ? "ready" : "waiting";
	}
	get queuedEdits(): boolean {
		return this.#training?.status === "training" && this.#revision > this.#training.training_set.label_seq;
	}
	/** Count only decoded, published chunks from the current model and current view. */
	get progress(): { ready: number; total: number } {
		const model = this.model;
		return { ready: model ? this.wanted.filter((key) => this.regions.get(key)?.model_id === model.id).length : 0, total: this.wanted.length };
	}
	get predictionPhase(): PredictionPhase {
		if (!this.model || !this.wanted.length) return "waiting";
		const progress = this.progress;
		if (progress.ready === progress.total) return "ready";
		if (this.predictionError) return "retrying";
		return this.#predictionBusy ? "predicting" : "queued";
	}

	constructor(private project: string, private image: string, private shape: Vec3,
		private pool: WorkerPool, private me: string, freeze: () => boolean) {
		this.#freeze = freeze;
		this.#timer = setInterval(() => this.pulse(), 500);
	}

	changed(revision: number) {
		if (revision <= this.#revision) return;
		this.#revision = revision;
		this.#dirty = true;
		this.#changedAt = Date.now();
	}

	async select(plugin: ModelPlugin) {
		const generation = ++this.#generation;
		this.#selecting = true;
		this.#loadingRegion?.keepOnly(new Set());
		this.store?.keepOnly(new Set());
		this.plugin = plugin;
		this.model = null;
		this.regions = new Map();
		this.store = null;
		this.#training = null;
		this.#dirty = true;
		this.#learningRetryAt = this.#predictionRetryAt = 0;
		this.#coveredRevision = -1;
		this.#lastTrain = 0;
		this.learningError = this.predictionError = "";
		this.learning = "Checking for previous examples…";
		this.predicting = "Waiting to start.";
		try {
			const models = await api<LiveModel[]>(`/api/projects/${this.project}/models`, { signal: this.#controller.signal });
			if (generation !== this.#generation || this.#stopped) return;
			this.learningError = "";
			this.#training = models.find((m) => m.live && m.created_by === this.me && m.status === "training") ?? null;
			this.model = models.find((m) => m.plugin === plugin.name && m.status === "ready" && m.training_set.image_artifact_id === this.image) ?? null;
			this.learning = this.model ? "Ready to use previously learned examples." : "Paint examples, including Background, to get started.";
		} catch (error) {
			if (!this.#stopped && generation === this.#generation) {
				this.learningError = message(error);
				this.learning = "Could not load what the app learned earlier.";
			}
		} finally { if (generation === this.#generation) this.#selecting = false; }
	}

	stop() {
		this.#stopped = true;
		this.#generation++;
		clearInterval(this.#timer);
		this.#controller.abort();
		this.#loadingRegion?.keepOnly(new Set());
		this.store?.keepOnly(new Set());
	}

	/** Retry learning without hiding usable suggestions or discarding their provenance. */
	retryLearning() {
		this.#dirty = true;
		this.#learningRetryAt = 0;
		this.learningError = "";
		this.learning = "Waiting to try learning again…";
	}

	private pulse() {
		if (this.#stopped || !this.enabled || this.paused || document.hidden || this.#freeze()) return;
		if (!this.#learningBusy && Date.now() >= this.#learningRetryAt) void this.learn();
		if (!this.#predictionBusy && Date.now() >= this.#predictionRetryAt) void this.predict();
	}

	private async learn() {
		const plugin = this.plugin;
		if (!plugin) return;
		const caps = plugin.capabilities;
		if (!this.#training && caps.learning === "manual") {
			this.learning = "To learn from new labels, use the Models page.";
			return;
		}
		if (!this.#training && (!this.#dirty || Date.now() - this.#changedAt < caps.debounce_ms || Date.now() - this.#lastTrain < caps.min_train_interval_ms)) return;
		this.#learningBusy = true;
		const generation = this.#generation;
		const revision = this.#revision;
		try {
			let trained: LiveModel;
			if (this.#training) {
				trained = await api<LiveModel>(`/api/projects/${this.project}/models/${this.#training.id}`, { signal: this.#controller.signal });
			} else {
				this.#lastTrain = Date.now();
				trained = await api<LiveModel>(`/api/projects/${this.project}/models`, { body: { plugin: plugin.name, live: true }, signal: this.#controller.signal });
				if (generation === this.#generation && trained.live_current) this.#coveredRevision = revision;
			}
			if (generation !== this.#generation || this.#stopped) return;
			this.learningError = "";
			this.#training = trained.status === "training" ? trained : null;
			this.#dirty = trained.plugin !== plugin.name || this.#coveredRevision < this.#revision;
			if (trained.status === "training") this.learning = this.#dirty ? "Learning from your saved labels… Newer edits will be used next." : "Learning from your saved labels…";
			else if (trained.status === "failed") {
				this.learning = "Could not learn from your latest labels.";
				this.learningError = trained.error || "Training failed. Add annotations or train explicitly on the Models page.";
			} else if (trained.plugin === plugin.name && trained.training_set.image_artifact_id === this.image) {
				this.learning = this.#dirty ? "Ready. Your newer labels will be used next." : "Finished learning from your saved labels.";
				// Don't change the displayed artifact set while an accept is being prepared.
				if (this.#freeze()) { this.#training = trained; return; }
				if (this.model?.id !== trained.id) {
					this.model = trained;
					// Retain the displayed chunks and their original provenance.
					// Inference replaces each one only after its bytes are decoded.
				}
			}
		} catch (error) {
			if (generation === this.#generation && !this.#stopped) {
				this.learningError = message(error);
				this.learning = "Could not learn from your latest labels.";
				// Do not loop on insufficient labels/quota. A saved edit or re-enable retries.
				this.#dirty = this.#revision > revision;
				this.#learningRetryAt = Date.now() + 5000;
			}
		} finally { this.#learningBusy = false; }
	}

	private async predict() {
		const model = this.model;
		const key = this.wanted.find((id) => this.regions.get(id)?.model_id !== model?.id);
		if (!model || !key) { this.predicting = model ? "Suggestions near your view are ready." : "Suggestions will appear after the app learns from your labels."; return; }
		this.#predictionBusy = true;
		const generation = this.#generation;
		this.predicting = `Adding suggestions nearby… ${this.wanted.filter((id) => this.regions.get(id)?.model_id === model.id).length} of ${this.wanted.length} areas ready.`;
		try {
			const result = await api<ChunkResult>(`/api/projects/${this.project}/models/${model.id}/live-chunk`, {
				body: { image_artifact_id: this.image, key: key.split("/").map(Number) }, signal: this.#controller.signal,
			});
			if (!result.ready) {
				// Poll only this one bounded job; moving the view replaces pending intent.
				let pipeline: Pipeline;
				do {
					await new Promise<void>((resolve) => setTimeout(resolve, 750));
					if (this.#stopped) return;
					pipeline = await api<Pipeline>(`/api/projects/${this.project}/pipelines/${result.pipeline_id}`, { signal: this.#controller.signal });
				} while (["waiting", "running"].includes(pipeline.status));
				if (pipeline.status !== "succeeded") throw new Error(pipeline.error || "Preview stopped; retrying shortly.");
			}
			if (this.#stopped || generation !== this.#generation || this.model?.id !== model.id || !this.wanted.includes(key)) return;
			const region: LiveRegion = { ...result, model_id: model.id, model_name: model.name, label_seq: model.training_set.label_seq,
				store: new ChunkStore(labelLoader(this.pool, absolute(result.zarr_url), this.shape), 1024 * 1024, 1) };
			this.#loadingRegion = region.store;
			// An artifact being ready is not enough: hold the old display until
			// the replacement's bytes are here, including slow/failed downloads.
			await region.store.request(key);
			// A finished chunk can wait locally while accepting; no new server work starts.
			while (this.#freeze() && !this.#stopped) await new Promise<void>((resolve) => setTimeout(resolve, 100));
			if (this.#stopped || generation !== this.#generation || this.model?.id !== model.id || !this.wanted.includes(key)) return;
			const regions = new Map(this.regions);
			regions.set(key, region);
			// Bound retained regions, not just queued work.
			for (const id of regions.keys()) { if (regions.size <= 256) break; if (!this.wanted.includes(id)) regions.delete(id); }
			// Snapshot the routing map. Acceptance can retain this exact immutable view.
			const store = new ChunkStore(async (id) => {
				const found = regions.get(id);
				if (found) return found.store.request(id);
				const origin = id.split("/").map((n) => Number(n) * 64);
				const shape = this.shape.map((n, i) => Math.min(64, n - origin[i]!));
				return { data: new Uint8Array(shape.reduce((a, b) => a * b, 1)), shape };
			}, 64 * 1024 * 1024, 4);
			// Warm the new routing snapshot without network reads so unaffected
			// chunks never blink or need decoding again during a local swap.
			await Promise.all([...regions].filter(([id, r]) => r.store.peek(id)).map(([id]) => store.request(id)));
			while (this.#freeze() && !this.#stopped) await new Promise<void>((resolve) => setTimeout(resolve, 100));
			if (this.#stopped || generation !== this.#generation || this.model?.id !== model.id || !this.wanted.includes(key)) return;
			this.regions = regions;
			this.store = store;
			this.predictionError = "";
		} catch (error) {
			if (!this.#stopped && generation === this.#generation && this.model?.id === model.id) { this.predictionError = message(error); this.#predictionRetryAt = Date.now() + 5000; }
		} finally { this.#loadingRegion = null; this.#predictionBusy = false; }
	}
}
