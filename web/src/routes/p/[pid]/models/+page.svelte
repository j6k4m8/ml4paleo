<script lang="ts">
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import type { Pipeline } from "#lib/types.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";

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
	// Prediction pipelines started from this page, by model.
	let predicting: Record<string, string> = $state({});
	let plugin = $state("rf");
	let params: Record<string, number> = $state({});
	let name = $state("");
	let error = $state("");
	let busy = $state(false);

	const chosen = $derived(plugins.find((p) => p.name === plugin));
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

	// Follow running trainings; refresh when one ends.
	$effect(() => {
		const ids = training ? training.split(",") : [];
		const sources = ids.map((id) => {
			const source = new EventSource(`/api/projects/${pid}/pipelines/${id}/events`);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				progress = { ...progress, [id]: update.progress };
				if (update.status !== "waiting" && update.status !== "running") {
					predicting = Object.fromEntries(Object.entries(predicting).filter(([, p]) => p !== id));
					refresh();
				}
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
			error =
				e instanceof ApiError && e.detail === "trained_model_quota_exceeded"
					? "You keep as many models as your quota allows. Delete one, or ask for more on your account page."
					: message(e);
		} finally {
			busy = false;
		}
	}

	async function predict(model: Model) {
		error = "";
		try {
			const started = await api<{ pipeline_id: string }>(`/api/projects/${pid}/models/${model.id}/predict`, {
				method: "POST",
			});
			predicting = { ...predicting, [model.id]: started.pipeline_id };
		} catch (e) {
			error = message(e);
		}
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

<p><a href="/p/{pid}">← Project</a> · <a href="/p/{pid}/annotate">Annotator</a> · <a href="/p/{pid}/rois">ROIs</a></p>
<h1>Models</h1>

<section>
	<h2>Train a model</h2>
	<p class="muted">
		Trains on everything labeled so far: complete ROIs (where unlabeled voxels count as background), open ROIs, and labels
		outside ROIs. ROIs marked validation are held out and used to score the model.
		{#if quota && quota.trained_models_limit !== null}
			You keep {quota.trained_models_used} of {quota.trained_models_limit} models.
		{/if}
	</p>
	<form class="stack" onsubmit={train}>
		{#if plugins.length > 1}
			<label>
				Kind
				<select bind:value={plugin}>
					{#each plugins as p (p.name)}<option value={p.name}>{p.name}</option>{/each}
				</select>
			</label>
		{/if}
		<label>Name (optional) <input bind:value={name} maxlength="100" placeholder="rf model" /></label>
		{#if chosen}
			<details>
				<summary>Settings</summary>
				{#each Object.entries(chosen.params_schema.properties) as [key, spec] (key)}
					<label>
						{spec.title ?? key}
						<input
							type="number"
							step={spec.type === "integer" ? 1 : "any"}
							min={spec.minimum}
							max={spec.maximum}
							bind:value={params[key]}
						/>
					</label>
				{/each}
			</details>
		{/if}
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		<button disabled={busy}>Train</button>
	</form>
</section>

<section>
	<h2>Trained</h2>
	{#if models.length === 0}
		<p class="muted">No models yet.</p>
	{:else}
		<ul class="models">
			{#each models as model (model.id)}
				<li class="card status-{model.status}">
					<div class="head">
						<strong>{model.name}</strong>
						<span class="badge">
							{model.status}{#if prediction?.model_id === model.id} · its prediction shows in the annotator{/if}
						</span>
						{#if model.status === "ready"}
							<button disabled={!!predicting[model.id]} onclick={() => predict(model)}>
								{predicting[model.id] ? "Predicting…" : "Predict"}
							</button>
						{/if}
						<button class="secondary" onclick={() => remove(model)}>Delete</button>
					</div>
					{#if model.status === "training"}
						<progress max="1" value={progress[model.pipeline_id ?? ""] ?? 0}></progress>
					{:else if predicting[model.id]}
						<progress max="1" value={progress[predicting[model.id] ?? ""] ?? 0}></progress>
					{/if}
					<p class="muted">
						{model.plugin} · {new Date(model.created_at).toLocaleString()} · trained on
						{model.training_set.labeled_chunks ?? 0} labeled chunks,
						{model.training_set.rois?.complete ?? 0} complete and {model.training_set.rois?.open ?? 0} open ROIs
					</p>
					{#if model.metrics}
						{#if model.metrics.validation_crops}
							<table>
								<thead><tr><th>Class</th><th>Dice</th><th>IoU</th></tr></thead>
								<tbody>
									{#each Object.entries(model.metrics.classes ?? {}) as [value, score] (value)}
										<tr><td>{className(value)}</td><td>{percent(score.dice)}</td><td>{percent(score.iou)}</td></tr>
									{/each}
									<tr><td>Mean</td><td>{percent(model.metrics.mean_dice)}</td><td></td></tr>
								</tbody>
							</table>
						{:else}
							<p class="muted">No validation ROIs, so no scores. Mark some ROIs as validation to score models.</p>
						{/if}
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</section>

<style>
	.models {
		list-style: none;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}
	.card {
		border: 1px solid var(--line);
		border-left: 4px solid var(--status);
		border-radius: 4px;
		padding: 0.5rem 0.75rem;
		background: var(--panel);
	}
	.status-training {
		--status: #e3b341;
	}
	.status-ready {
		--status: #57ab5a;
	}
	.status-failed {
		--status: #e5534b;
	}
	.head {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	.head button {
		padding: 0.2rem 0.6rem;
	}
	.head .badge {
		margin-right: auto;
	}
	.badge {
		font-size: 0.8rem;
		color: var(--muted);
	}
	table {
		border-collapse: collapse;
	}
	th,
	td {
		text-align: left;
		padding: 0.15rem 1rem 0.15rem 0;
	}
	.card p {
		margin: 0.3rem 0;
	}
</style>
