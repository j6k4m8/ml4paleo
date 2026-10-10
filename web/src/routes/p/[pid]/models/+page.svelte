<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { trainingNotifications, type TrainingModel } from "#lib/training-notifications.svelte.ts";
	import { latestPredictions, unfinished } from "#lib/pipelines.ts";
	import type { Pipeline } from "#lib/types.ts";
	import type { ModelPlugin } from "#lib/viewer/live.ts";
	import { BACKGROUND_VALUE, withBackground } from "#lib/viewer/background.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";
	import { SHOW_ROIS } from "#lib/features.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Labeled from "#lib/ui/Labeled.svelte";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import Brain from "@lucide/svelte/icons/brain";
	import Play from "@lucide/svelte/icons/play";
	import Trash from "@lucide/svelte/icons/trash";

	interface Plugin extends ModelPlugin {
		name: string;
		version: string;
		devices: string[];
		params_schema: { properties: Record<string, { type: string; default: number; minimum?: number; maximum?: number; title?: string }> };
	}

	interface Model extends TrainingModel {
		live: boolean;
		id: string;
		name: string;
		plugin: string;
		status: "training" | "ready" | "failed";
		/** Why training failed, if it says. */
		error?: string | null;
		params: Record<string, number>;
		class_values: number[];
		metrics: {
			mean_dice?: number | null;
			mean_iou?: number | null;
			accuracy?: number | null;
			voxels?: number;
			evaluation?: "human_validation_rois" | "human_out_of_bag";
			evaluation_voxels?: number;
			validation_crops?: number;
			classes?: Record<string, { dice: number; iou: number; voxels: number }>;
		} | null;
		pipeline_id: string | null;
		training_set: { label_seq?: number; labeled_chunks?: number; rois?: { complete: number; open: number; validation: number } };
		created_at: string;
	}

	interface Quota {
		trained_models_limit: number | null;
		trained_models_used: number;
	}

	const pid = $derived(page.params.pid ?? "");
	let models: Model[] = $state([]);
	let loadedProject = $state("");
	const targetModel = $derived(page.url.hash.startsWith("#model-") ? page.url.hash.slice(7) : null);
	const missingModel = $derived(loadedProject === pid && targetModel && !models.some((model) => model.id === targetModel));
	let plugins: Plugin[] = $state([]);
	let classes: LabelClass[] = $state([]);
	let classesLoaded = $state(false);
	// Voxels labeled with each value (background is 1), once known.
	let counts: Record<string, number> | null = $state(null);
	// Whether the project has any ROIs, once known. They're hidden now, but a complete one
	// made earlier makes the voxels in it background, which the counts don't show.
	let hasRois: boolean | null = $state(null);
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
	let refreshGeneration = 0;
	const trainBusy = $derived(trainingNotifications.isTraining(pid));

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Models" }]);
	});

	$effect(() => {
		api<{ name: string }>(`/api/projects/${pid}`).then((p) => (projectName = p.name), () => {});
	});

	const chosen = $derived(plugins.find((p) => p.name === plugin));
	// What has been labeled, background and classes, and so whether there are two things to tell apart.
	const labeledNow = $derived(withBackground(classes).filter((c) => (counts?.[c.value] ?? 0) > 0));
	const tooLittle = $derived(counts !== null && classesLoaded && labeledNow.length < 2);
	// Training can't work: nothing painted would make up for it, so Train waits. With ROIs around
	// it only warns, and the trainer says if it really can't.
	const blocked = $derived(tooLittle && hasRois === false);
	const whyTooLittle = $derived.by(() => {
		const only = labeledNow[0];
		if (!only) return "Nothing is labeled yet. In the annotator, label what you're looking for, and some background.";
		if (only.value === BACKGROUND_VALUE) return "Only background is labeled. Label what you're looking for too.";
		return `Only ${only.name} is labeled. Paint some Background too, so the model can tell them apart.`;
	});
	// The prediction pipelines still running, by model.
	const predicting: Record<string, string> = $derived(
		Object.fromEntries(Object.entries(predictions).flatMap(([model, p]) => (unfinished(p) ? [[model, p.id]] : []))),
	);
	const training = $derived(
		[...models.filter((m) => m.status === "training").map((m) => m.pipeline_id), ...Object.values(predicting)].join(","),
	);
	const readyModels = $derived(models.filter((model) => model.status === "ready"));
	const latestScoredSeq = $derived(
		Math.max(
			-1,
			...readyModels
				.filter((model) => scoreVoxels(model) > 0 && model.training_set.label_seq !== undefined)
				.map((model) => model.training_set.label_seq ?? -1),
		),
	);
	const bestCurrentDice = $derived(
		Math.max(
			-1,
			...readyModels
				.filter((model) => model.training_set.label_seq === latestScoredSeq)
				.map((model) => model.metrics?.mean_dice ?? -1),
		),
	);

	async function refresh() {
		const project = pid, generation = ++refreshGeneration;
		try {
			const [found, limits, predicted, pipelines, labeled, rois] = await Promise.all([
				api<Model[]>(`/api/projects/${project}/models`),
				api<Quota>("/api/me/quota").catch(() => null),
				api<{ model_id: string | null; model_name: string | null }>(`/api/projects/${project}/prediction`).catch(() => null),
				api<Pipeline[]>(`/api/projects/${project}/pipelines`),
				api<Record<string, number>>(`/api/projects/${project}/labels/counts`).catch(() => null),
				api<unknown[]>(`/api/projects/${project}/rois`).then((list) => list.length > 0, () => null),
			]);
			if (project !== pid || generation !== refreshGeneration) return;
			models = found;
			loadedProject = project;
			for (const model of found) trainingNotifications.watch(project, model);
			quota = limits;
			prediction = predicted;
			predictions = latestPredictions(pipelines);
			counts = labeled;
			hasRois = rois;
		} catch (e) {
			if (project === pid && generation === refreshGeneration) error = message(e);
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
		api<LabelClass[]>(`/api/projects/${pid}/labels/classes`).then(
			(list) => {
				classes = list;
				classesLoaded = true;
			},
			() => {},
		);
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
		if (busy || blocked || trainBusy) return;
		const project = pid;
		error = "";
		try {
			const model = await trainingNotifications.submit(project, { plugin, params: { ...params }, name: name || undefined });
			if (!model || project !== pid) return;
			name = "";
			await refresh();
		} catch (e) {
			if (project === pid) error = message(e);
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

	function scoreVoxels(model: Model): number {
		if (model.metrics?.evaluation_voxels !== undefined) return model.metrics.evaluation_voxels;
		return model.metrics?.validation_crops ? (model.metrics.voxels ?? 0) : 0;
	}

	function evaluationName(model: Model): string {
		switch (model.metrics?.evaluation) {
			case "human_validation_rois":
				return "Held-out validation ROIs";
			case "human_out_of_bag":
				return "Out-of-bag human labels";
			default:
				return scoreVoxels(model) > 0 ? "Legacy validation" : "Not graded";
		}
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-[20rem_1fr]">
	<section class="panel self-start">
		<h2 class="panel-title">Train a model</h2>
		<form class="flex flex-col gap-3 p-3" onsubmit={train}>
			<p class="text-ink-dim">
				{#if SHOW_ROIS}
					Trains on everything labeled so far: complete ROIs (unlabeled voxels there count as background), open ROIs, and
					labels outside ROIs. Validation ROIs are held out to score the model.
				{:else}
					Trains on everything you've labeled: your classes, and Background, which tells the model what to leave alone. Each model is
					graded on held-out human-drawn or imported voxels (out-of-bag for RF); accepted predictions never grade the model.
				{/if}
			</p>
			{#if plugins.length > 1}
				<label class="label">
					Kind
					<select class="field" bind:value={plugin}>
						{#each plugins as p (p.name)}<option value={p.name}>{p.capabilities.display_name}</option>{/each}
					</select>
				</label>
			{/if}
			<label class="label">Name (optional) <input class="field" bind:value={name} maxlength="100" placeholder="rf model" /></label>
			{#if chosen}
				<p class="text-2xs text-ink-dim">{chosen.capabilities.display_name} · {chosen.capabilities.family} · {chosen.devices.join(" / ")} · {chosen.capabilities.learning} learning in Live. Training here creates a saved checkpoint.</p>
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
			{#if counts && classesLoaded}
				<div class="flex flex-col gap-1.5 rounded-sm border border-edge bg-field p-2.5">
					<p class="text-2xs font-semibold tracking-wide text-ink-dim uppercase">Labeled so far</p>
					<Labeled {classes} {counts} />
					{#if tooLittle}
						<p class="text-warn" role="status">{whyTooLittle}</p>
						<a href="/p/{pid}/annotate">Open the annotator</a>
					{/if}
				</div>
			{/if}
			{#if error}<p class="error" role="alert">{error}</p>{/if}
			<button class="btn btn-primary h-7" disabled={busy || blocked || trainBusy} title={blocked ? whyTooLittle : undefined}><Brain size={14} /> {trainBusy ? "Training…" : "Train"}</button>
		</form>
	</section>

	<section class="flex flex-col gap-3">
		<h1>Models</h1>
		{#if missingModel}<p class="muted" role="status">This model is no longer available.</p>{/if}
		{#if models.length === 0}
			<div class="panel grid place-items-center gap-2 p-10 text-center">
				<Brain size={28} class="text-ink-faint" />
				<p class="muted">No models yet.</p>
			</div>
		{:else}
			{#if readyModels.length > 0}
			<section class="panel overflow-x-auto" aria-labelledby="comparison-title">
				<div class="panel-title flex items-center justify-between gap-3" id="comparison-title">
					<span>Model comparison</span>
					<span class="text-2xs font-normal tracking-normal text-ink-faint normal-case">Human labels only</span>
				</div>
				<table class="w-full min-w-[42rem] text-2xs">
					<thead class="text-ink-dim">
						<tr>
							<th class="px-3 py-2 text-left font-normal">Model</th>
							<th class="px-3 py-2 text-right font-normal">Mean Dice</th>
							<th class="px-3 py-2 text-right font-normal">Accuracy</th>
							<th class="px-3 py-2 text-right font-normal">Human voxels</th>
							<th class="px-3 py-2 text-right font-normal">Label state</th>
							<th class="px-3 py-2 text-left font-normal">Method</th>
						</tr>
					</thead>
					<tbody>
						{#each readyModels as model (model.id)}
							{@const current = model.training_set.label_seq === latestScoredSeq}
							{@const best = current && model.metrics?.mean_dice === bestCurrentDice && bestCurrentDice >= 0}
							<tr class="border-t border-edge {current ? '' : 'text-ink-dim'}">
								<td class="px-3 py-2 font-medium text-ink">{model.name}{#if best}<span class="ml-1.5 text-ok">best</span>{/if}</td>
								<td class="px-3 py-2 text-right font-mono">{percent(model.metrics?.mean_dice)}</td>
								<td class="px-3 py-2 text-right font-mono">{percent(model.metrics?.accuracy)}</td>
								<td class="px-3 py-2 text-right font-mono">{scoreVoxels(model).toLocaleString()}</td>
								<td class="px-3 py-2 text-right font-mono">#{model.training_set.label_seq ?? "–"}</td>
								<td class="px-3 py-2">{evaluationName(model)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
				<p class="border-t border-edge px-3 py-2 text-2xs text-ink-dim">
					“Best” compares models trained from the newest shared label state. Older rows remain useful history, but their test voxels may differ.
				</p>
			</section>
			{/if}
			<ul class="flex flex-col gap-2">
				{#each models as model (model.id)}
					{@const last = predictions[model.id]}
					<li
						id="model-{model.id}"
						tabindex="-1"
						{@attach (node) => {
							// Models arrive after navigation; the browser's initial hash jump is too early.
							if (targetModel === model.id) {
								node.scrollIntoView({ block: "start" });
								node.focus({ preventScroll: true });
							}
						}}
						class="panel flex scroll-mt-3 flex-col gap-2 border-l-2 p-3 target:outline target:outline-accent
							{model.status === 'ready' ? 'border-l-ok' : model.status === 'failed' ? 'border-l-danger' : 'border-l-warn'}"
					>
						<div class="flex items-center gap-2">
						<span class="font-medium">{model.name}</span>
						{#if model.live}<span class="text-2xs text-ink-dim">Live · rolling checkpoint</span>{/if}
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
						{#if model.status === "failed"}
							<p class="error text-2xs" role="alert">Training failed{model.error ? `: ${model.error}` : "."}</p>
						{/if}
						{#if last?.status === "failed"}
							<p class="error text-2xs" role="alert">The last prediction failed{last.error ? `: ${last.error}` : "."}</p>
						{/if}
						<p class="text-2xs text-ink-faint">
							{model.plugin} · {new Date(model.created_at).toLocaleString()} · {model.training_set.labeled_chunks ?? 0} labeled
							chunks{#if SHOW_ROIS}, {model.training_set.rois?.complete ?? 0} complete and {model.training_set.rois?.open ?? 0} open ROIs{/if}
						</p>
						{#if model.metrics}
							{#if scoreVoxels(model) > 0}
								<p class="text-2xs text-ink-dim">
									{evaluationName(model)} · {scoreVoxels(model).toLocaleString()} human-annotated voxels
								</p>
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
								<p class="text-2xs text-ink-dim">No held-out human labels were available to grade this model.</p>
							{/if}
						{/if}
					</li>
				{/each}
			</ul>
		{/if}
	</section>
</div>
