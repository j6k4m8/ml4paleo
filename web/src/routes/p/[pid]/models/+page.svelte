<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { latestPredictions, unfinished } from "#lib/pipelines.ts";
	import type { Pipeline } from "#lib/types.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import Brain from "@lucide/svelte/icons/brain";
	import Play from "@lucide/svelte/icons/play";
	import Trash from "@lucide/svelte/icons/trash";

	interface Plugin {
		name: string;
		version: string;
		devices: string[];
		params_schema: { properties: Record<string, { type: string; default: number; minimum?: number; maximum?: number; title?: string }> };
	}

	interface Model {
		id: string;
		name: string;
		plugin: string;
		status: "training" | "ready" | "failed";
		params: Record<string, number>;
		class_values: number[];
		metrics: {
			mean_dice?: number | null;
			accuracy?: number | null;
			validation_crops?: number;
			classes?: Record<string, { dice: number; iou: number; voxels: number }>;
		} | null;
		pipeline_id: string | null;
		training_set: { labeled_chunks?: number; rois?: { complete: number; open: number; validation: number } };
		created_at: string;
	}

	interface Quota {
		trained_models_limit: number | null;
		trained_models_used: number;
	}

	const pid = $derived(page.params.pid ?? "");
	let models: Model[] = $state([]);
	let plugins: Plugin[] = $state([]);
	let classes: LabelClass[] = $state([]);
	let quota: Quota | null = $state(null);
	let progress: Record<string, number> = $state({});
	let prediction: { model_id: string | null; model_name: string | null } | null = $state(null);
	// Each model's latest prediction pipeline.
	let predictions: Record<string, Pipeline> = $state({});
	let plugin = $state("rf");
	let params: Record<string, number> = $state({});
	let name = $state("");
	let error = $state("");
	let busy = $state(false);
	let projectName = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Models" }]);
	});

	$effect(() => {
		api<{ name: string }>(`/api/projects/${pid}`).then((p) => (projectName = p.name), () => {});
	});

	const chosen = $derived(plugins.find((p) => p.name === plugin));
	// The prediction pipelines still running, by model.
	const predicting: Record<string, string> = $derived(
		Object.fromEntries(Object.entries(predictions).flatMap(([model, p]) => (unfinished(p) ? [[model, p.id]] : []))),
	);
	const training = $derived(
		[...models.filter((m) => m.status === "training").map((m) => m.pipeline_id), ...Object.values(predicting)].join(","),
	);

	async function refresh() {
		try {
			models = await api<Model[]>(`/api/projects/${pid}/models`);
			quota = await api<Quota>("/api/me/quota").catch(() => null);
			prediction = await api<{ model_id: string | null; model_name: string | null }>(
				`/api/projects/${pid}/prediction`,
			).catch(() => null);
			predictions = latestPredictions(await api<Pipeline[]>(`/api/projects/${pid}/pipelines`));
		} catch (e) {
			error = message(e);
		}
	}

	$effect(() => {
		if (!pid) return;
		refresh();
		api<Plugin[]>("/api/plugins").then((list) => {
			plugins = list;
			const first = list[0];
			if (first) params = Object.fromEntries(Object.entries(first.params_schema.properties).map(([k, v]) => [k, v.default]));
		}, () => {});
		api<LabelClass[]>(`/api/projects/${pid}/labels/classes`).then((list) => (classes = list), () => {});
	});

	// Follow running trainings and predictions; refresh when one ends. For a
	// pipeline that has already ended the server answers 204, which closes
	// the stream without a status event.
	$effect(() => {
		const ids = training ? training.split(",") : [];
		const sources = ids.map((id) => {
			const source = new EventSource(`/api/projects/${pid}/pipelines/${id}/events`);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				progress = { ...progress, [id]: update.progress };
				if (!unfinished(update)) {
					source.close();
					refresh();
				}
			});
			source.addEventListener("error", () => {
				if (source.readyState === EventSource.CLOSED) refresh();
			});
			return source;
		});
		return () => sources.forEach((source) => source.close());
	});

	async function train(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await api(`/api/projects/${pid}/models`, { body: { plugin, params, name: name || undefined } });
			name = "";
			await refresh();
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}

	async function predict(model: Model) {
		busy = true;
		error = "";
		try {
			await api(`/api/projects/${pid}/models/${model.id}/predict`, { method: "POST" });
		} catch (e) {
			error = message(e);
		}
		// Shows the new prediction running, or the one already running if
		// this was refused.
		await refresh();
		busy = false;
	}

	async function remove(model: Model) {
		if (!confirm(`Delete ${model.name}?`)) return;
		try {
			await api(`/api/projects/${pid}/models/${model.id}`, { method: "DELETE" });
			await refresh();
		} catch (e) {
			error = message(e);
		}
	}

	function className(value: string): string {
		return classes.find((c) => String(c.value) === value)?.name ?? `class ${value}`;
	}

	function percent(value: number | null | undefined): string {
		return value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`;
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-[20rem_1fr]">
	<section class="panel self-start">
		<h2 class="panel-title">Train a model</h2>
		<form class="flex flex-col gap-3 p-3" onsubmit={train}>
			<p class="text-ink-dim">
				Trains on everything labeled so far: complete ROIs (unlabeled voxels there count as background), open ROIs, and
				labels outside ROIs. Validation ROIs are held out to score the model.
			</p>
			{#if plugins.length > 1}
				<label class="label">
					Kind
					<select class="field" bind:value={plugin}>
						{#each plugins as p (p.name)}<option value={p.name}>{p.name}</option>{/each}
					</select>
				</label>
			{/if}
			<label class="label">Name (optional) <input class="field" bind:value={name} maxlength="100" placeholder="rf model" /></label>
			{#if chosen}
				<details class="group rounded-sm border border-edge bg-field">
					<summary class="cursor-pointer px-2 py-1 text-2xs text-ink-dim select-none hover:text-ink">Settings</summary>
					<div class="grid grid-cols-2 gap-2 p-2">
						{#each Object.entries(chosen.params_schema.properties) as [key, spec] (key)}
							<label class="label">
								{spec.title ?? key}
								<input
									class="field font-mono"
									type="number"
									step={spec.type === "integer" ? 1 : "any"}
									min={spec.minimum}
									max={spec.maximum}
									bind:value={params[key]}
								/>
							</label>
						{/each}
					</div>
				</details>
			{/if}
			{#if quota && quota.trained_models_limit !== null}
				<div class="flex flex-col gap-1">
					<div class="flex justify-between text-2xs text-ink-dim">
						<span>Models kept</span><span class="font-mono">{quota.trained_models_used} / {quota.trained_models_limit}</span>
					</div>
					<div class="h-1 overflow-hidden rounded-full bg-field">
						<div class="h-full bg-accent" style:width="{Math.min(100, (100 * quota.trained_models_used) / Math.max(1, quota.trained_models_limit))}%"></div>
					</div>
				</div>
			{/if}
			{#if error}<p class="error" role="alert">{error}</p>{/if}
			<button class="btn btn-primary h-7" disabled={busy}><Brain size={14} /> Train</button>
		</form>
	</section>

	<section class="flex flex-col gap-3">
		<h1>Models</h1>
		{#if models.length === 0}
			<div class="panel grid place-items-center gap-2 p-10 text-center">
				<Brain size={28} class="text-ink-faint" />
				<p class="muted">No models yet.</p>
			</div>
		{:else}
			<ul class="flex flex-col gap-2">
				{#each models as model (model.id)}
					{@const last = predictions[model.id]}
					<li
						class="panel flex flex-col gap-2 border-l-2 p-3
							{model.status === 'ready' ? 'border-l-ok' : model.status === 'failed' ? 'border-l-danger' : 'border-l-warn'}"
					>
						<div class="flex items-center gap-2">
							<span class="font-medium">{model.name}</span>
							<span class="rounded-sm bg-field px-1.5 py-0.5 text-2xs text-ink-dim">{model.status}</span>
							{#if prediction?.model_id === model.id}
								<span class="rounded-sm bg-accent-soft px-1.5 py-0.5 text-2xs text-ink">shown in the annotator</span>
							{/if}
							<div class="ml-auto flex gap-1">
								{#if model.status === "ready"}
									<button class="btn" disabled={busy || !!predicting[model.id]} onclick={() => predict(model)}>
										<Play size={12} />
										{predicting[model.id] ? "Predicting…" : "Predict"}
									</button>
								{/if}
								<button class="btn btn-ghost btn-danger" onclick={() => remove(model)} aria-label="Delete {model.name}">
									<Trash size={13} />
								</button>
							</div>
						</div>
						{#if model.status === "training"}
							<progress class="h-1 w-full" max="1" value={progress[model.pipeline_id ?? ""] ?? 0}></progress>
						{:else if last && predicting[model.id]}
							<progress class="h-1 w-full" max="1" value={progress[last.id] ?? last.progress}></progress>
						{/if}
						{#if last?.status === "failed"}
							<p class="error text-2xs" role="alert">The last prediction failed{last.error ? `: ${last.error}` : "."}</p>
						{/if}
						<p class="text-2xs text-ink-faint">
							{model.plugin} · {new Date(model.created_at).toLocaleString()} · {model.training_set.labeled_chunks ?? 0} labeled
							chunks, {model.training_set.rois?.complete ?? 0} complete and {model.training_set.rois?.open ?? 0} open ROIs
						</p>
						{#if model.metrics}
							{#if model.metrics.validation_crops}
								<table class="w-full max-w-md text-2xs">
									<thead class="text-ink-dim">
										<tr><th class="py-1 text-left font-normal">Class</th><th class="text-right font-normal">Dice</th><th class="text-right font-normal">IoU</th></tr>
									</thead>
									<tbody class="font-mono">
										{#each Object.entries(model.metrics.classes ?? {}) as [value, score] (value)}
											<tr class="border-t border-edge">
												<td class="py-1 font-sans">{className(value)}</td>
												<td class="text-right">{percent(score.dice)}</td>
												<td class="text-right">{percent(score.iou)}</td>
											</tr>
										{/each}
										<tr class="border-t border-edge font-semibold">
											<td class="py-1 font-sans">Mean</td><td class="text-right">{percent(model.metrics.mean_dice)}</td><td></td>
										</tr>
									</tbody>
								</table>
							{:else}
								<p class="text-2xs text-ink-dim">No validation ROIs, so no scores. Mark some ROIs as validation to score models.</p>
							{/if}
						{/if}
					</li>
				{/each}
			</ul>
		{/if}
	</section>
</div>
