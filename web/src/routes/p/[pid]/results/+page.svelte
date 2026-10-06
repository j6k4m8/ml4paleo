<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import Combine from "@lucide/svelte/icons/combine";
	import Shapes from "@lucide/svelte/icons/shapes";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import type { Pipeline } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";

	interface Prediction {
		model_name: string | null;
		committed_at: string;
	}

	interface Segmentation {
		model_name: string | null;
		min_voxels: number;
		label_seq: number;
		committed_at: string;
	}

	const pid = $derived(page.params.pid ?? "");
	let projectName = $state("");
	let prediction: Prediction | null = $state(null);
	let segmentation: Segmentation | null = $state(null);
	let loaded = $state(false);
	let minVoxels = $state(50);
	let running: string | null = $state(null);
	let progress = $state(0);
	let error = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Results" }]);
	});

	const missing = (e: unknown) => {
		if (e instanceof ApiError && e.status === 404) return null;
		throw e;
	};

	async function refresh() {
		try {
			projectName = (await api<{ name: string }>(`/api/projects/${pid}`)).name;
			prediction = await api<Prediction>(`/api/projects/${pid}/prediction`).catch(missing);
			segmentation = await api<Segmentation>(`/api/projects/${pid}/segmentation`).catch(missing);
		} catch (e) {
			error = message(e);
		} finally {
			loaded = true;
		}
	}

	$effect(() => {
		if (pid) refresh();
	});

	// Follow the pipeline this page started.
	$effect(() => {
		if (!running) return;
		const source = new EventSource(`/api/projects/${pid}/pipelines/${running}/events`);
		source.addEventListener("status", (event) => {
			const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
			progress = update.progress;
			if (update.status !== "waiting" && update.status !== "running") {
				if (update.status !== "succeeded") error = update.error ?? `The final segmentation ${update.status}.`;
				running = null;
				refresh();
			}
		});
		return () => source.close();
	});

	async function compose(event: SubmitEvent) {
		event.preventDefault();
		error = "";
		try {
			const started = await api<{ pipeline_id: string }>(`/api/projects/${pid}/segmentation`, {
				body: { min_voxels: minVoxels },
			});
			progress = 0;
			running = started.pipeline_id;
		} catch (e) {
			error = message(e);
		}
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-2">
	<section class="panel self-start">
		<h2 class="panel-title">Prediction</h2>
		<div class="flex flex-col gap-2 p-3">
			{#if prediction}
				<p>
					From <span class="font-medium">{prediction.model_name ?? "a deleted model"}</span>,
					<span class="text-ink-dim">{new Date(prediction.committed_at).toLocaleString()}</span>
				</p>
				<a class="btn self-start hover:no-underline" href="/p/{pid}/annotate"><Brush size={13} /> Review in the annotator</a>
			{:else if loaded}
				<p class="text-ink-dim">No prediction yet. Train a model and predict on the <a href="/p/{pid}/models">Models</a> page.</p>
			{/if}
		</div>
	</section>

	<section class="panel self-start">
		<h2 class="panel-title">Final segmentation</h2>
		<div class="flex flex-col gap-3 p-3">
			<p class="text-ink-dim">
				The prediction with your labels on top (unlabeled voxels in complete ROIs count as background), and specks of each
				class removed unless someone labeled part of them.
			</p>
			{#if segmentation}
				<div class="flex items-start gap-2 rounded-sm border border-edge bg-field p-2">
					<Shapes size={16} class="mt-0.5 text-ok" />
					<div class="flex flex-col gap-0.5">
						<span>Made {new Date(segmentation.committed_at).toLocaleString()}</span>
						<span class="text-2xs text-ink-dim">
							from {segmentation.model_name ?? "a deleted model"} · specks under {segmentation.min_voxels} voxels removed · labels up
							to edit {segmentation.label_seq}
						</span>
					</div>
				</div>
			{/if}
			<form class="flex items-end gap-2" onsubmit={compose}>
				<label class="label">
					Smallest piece kept (voxels)
					<input class="field w-32 font-mono" type="number" min="0" step="1" bind:value={minVoxels} />
				</label>
				<button class="btn btn-primary" disabled={!prediction || !!running}>
					<Combine size={13} />
					{running ? "Making…" : segmentation ? "Make it again" : "Make final segmentation"}
				</button>
			</form>
			{#if running}
				<div class="flex items-center gap-2">
					<progress class="h-1 flex-1 accent-accent" max="1" value={progress}></progress>
					<span class="font-mono text-2xs text-ink-dim">{Math.round(progress * 100)}%</span>
				</div>
			{/if}
			{#if error}<p class="error" role="alert">{error}</p>{/if}
		</div>
	</section>
</div>
