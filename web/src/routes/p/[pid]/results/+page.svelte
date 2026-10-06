<script lang="ts">
	import Box from "@lucide/svelte/icons/box";
	import Brush from "@lucide/svelte/icons/brush";
	import Combine from "@lucide/svelte/icons/combine";
	import Download from "@lucide/svelte/icons/download";
	import Shapes from "@lucide/svelte/icons/shapes";
	import { untrack } from "svelte";
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
		artifact_id: string;
		model_name: string | null;
		min_voxels: number;
		label_seq: number;
		committed_at: string;
	}

	type Format = "stl" | "obj" | "glb";

	interface Meshes {
		segmentation_artifact_id: string | null;
		info: {
			units: string;
			downsample: number;
			method: string;
			classes: { value: number; name: string; color: string; files: Record<Format, string> }[];
		};
		files_url: string;
		committed_at: string;
	}

	const FORMATS: [Format, string][] = [
		["stl", "STL"],
		["obj", "OBJ"],
		["glb", "GLB"],
	];

	const pid = $derived(page.params.pid ?? "");
	let projectName = $state("");
	let prediction: Prediction | null = $state(null);
	let segmentation: Segmentation | null = $state(null);
	let meshes: Meshes | null = $state(null);
	let pipelines: Pipeline[] = $state([]);
	let loaded = $state(false);
	let minVoxels = $state(50);
	let downsample = $state(1);
	let method = $state("any");
	let simplify = $state(1);
	let error = $state("");
	let composeError = $state("");
	let meshError = $state("");
	// Set while a request to start a pipeline is out, so a double click
	// doesn't start two.
	let starting = $state(false);

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Results" }]);
	});

	const missing = (e: unknown) => {
		if (e instanceof ApiError && e.status === 404) return null;
		throw e;
	};
	const active = (p: Pipeline | undefined) => p?.status === "waiting" || p?.status === "running";
	// The newest of each kind (the list is newest first).
	const composing = $derived(pipelines.find((p) => p.kind === "segmentation"));
	const meshing = $derived(pipelines.find((p) => p.kind === "meshes"));
	const stale = $derived.by(() => !!meshes && !!segmentation && meshes.segmentation_artifact_id !== segmentation.artifact_id);

	async function refresh() {
		try {
			projectName = (await api<{ name: string }>(`/api/projects/${pid}`)).name;
			[prediction, segmentation, meshes, pipelines] = await Promise.all([
				api<Prediction>(`/api/projects/${pid}/prediction`).catch(missing),
				api<Segmentation>(`/api/projects/${pid}/segmentation`).catch(missing),
				api<Meshes>(`/api/projects/${pid}/meshes`).catch(missing),
				api<Pipeline[]>(`/api/projects/${pid}/pipelines`),
			]);
		} catch (e) {
			error = message(e);
		} finally {
			loaded = true;
		}
	}

	$effect(() => {
		if (pid) refresh();
	});

	// Only which pipelines run, so status updates don't reopen the streams.
	const running = $derived(
		[composing, meshing]
			.filter(active)
			.map((p) => p?.id)
			.join(","),
	);

	// Follow the running pipelines; the server ends a stream (204) once its
	// pipeline is done, so refresh then too.
	$effect(() => {
		const ids = running ? running.split(",") : [];
		const sources = ids.map((id) => {
			const source = new EventSource(`/api/projects/${untrack(() => pid)}/pipelines/${id}/events`);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				pipelines = untrack(() => pipelines).map((p) => (p.id === update.id ? update : p));
				if (!active(update)) refresh();
			});
			source.addEventListener("error", () => {
				if (source.readyState === EventSource.CLOSED) refresh();
			});
			return source;
		});
		return () => sources.forEach((source) => source.close());
	});

	async function start(url: string, body: object, fail: (text: string) => void) {
		if (starting) return;
		starting = true;
		fail("");
		try {
			await api(url, { body });
		} catch (e) {
			fail(message(e));
		}
		await refresh();
		starting = false;
	}

	function compose(event: SubmitEvent) {
		event.preventDefault();
		start(`/api/projects/${pid}/segmentation`, { min_voxels: minVoxels }, (text) => (composeError = text));
	}

	function makeMeshes(event: SubmitEvent) {
		event.preventDefault();
		start(`/api/projects/${pid}/meshes`, { downsample, method, simplify }, (text) => (meshError = text));
	}

	const slug = (text: string) =>
		text
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-|-$/g, "") || "mesh";
</script>

{#snippet progress(pipeline: Pipeline | undefined, failed: string)}
	{#if pipeline && active(pipeline)}
		<div class="flex items-center gap-2">
			<progress class="h-1 flex-1 accent-accent" max="1" value={pipeline.progress}></progress>
			<span class="font-mono text-2xs text-ink-dim">{Math.round(pipeline.progress * 100)}%</span>
		</div>
	{:else if pipeline && pipeline.status !== "succeeded"}
		<p class="error" role="alert">{pipeline.error ?? `The last one ${pipeline.status}.`}</p>
	{/if}
	{#if failed}<p class="error" role="alert">{failed}</p>{/if}
{/snippet}

<ProjectTabs {pid} />

<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-2">
	{#if error}<p class="error lg:col-span-2" role="alert">{error}</p>{/if}

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
				<button class="btn btn-primary" disabled={!prediction || active(composing) || starting}>
					<Combine size={13} />
					{active(composing) ? "Making…" : segmentation ? "Make it again" : "Make final segmentation"}
				</button>
			</form>
			{@render progress(composing, composeError)}
		</div>
	</section>

	<section class="panel self-start lg:col-span-2">
		<h2 class="panel-title">Meshes</h2>
		<div class="flex flex-col gap-3 p-3">
			<p class="text-ink-dim">
				A surface for each class of the final segmentation, in x, y, z and the scan's units, for 3D printing or other tools.
			</p>
			{#if meshes}
				{#if stale}
					<p class="text-2xs text-warn">These are from an older final segmentation; make them again to match the current one.</p>
				{/if}
				<ul class="flex flex-col divide-y divide-edge rounded-sm border border-edge bg-field">
					{#each meshes.info.classes as mesh (mesh.value)}
						<li class="flex flex-wrap items-center gap-2 px-2 py-1.5">
							<span class="size-3 shrink-0 rounded-xs border border-edge" style:background={mesh.color}></span>
							<span class="min-w-24 flex-1 font-medium">{mesh.name}</span>
							{#each FORMATS as [format, label] (format)}
								<a
									class="btn btn-ghost hover:no-underline"
									href={meshes.files_url + mesh.files[format]}
									download="{slug(projectName)}-{slug(mesh.name)}.{format}"
								>
									<Download size={13} />
									{label}
								</a>
							{/each}
						</li>
					{:else}
						<li class="px-2 py-1.5 text-ink-dim">No class has any voxels in the final segmentation.</li>
					{/each}
				</ul>
				<p class="text-2xs text-ink-dim">
					Made {new Date(meshes.committed_at).toLocaleString()} · in {meshes.info.units}
					{#if meshes.info.downsample > 1}· at 1/{meshes.info.downsample} resolution{/if}
				</p>
			{/if}
			<form class="flex flex-wrap items-end gap-2" onsubmit={makeMeshes}>
				<label class="label">
					Resolution
					<select class="field" bind:value={downsample}>
						<option value={1}>Full</option>
						<option value={2}>1/2</option>
						<option value={4}>1/4</option>
						<option value={8}>1/8</option>
					</select>
				</label>
				<label class="label">
					Coarse voxels keep
					<select class="field" bind:value={method} disabled={downsample === 1}>
						<option value="any">thin parts</option>
						<option value="majority">a smoother surface</option>
					</select>
				</label>
				<label class="label">
					Simplify, moving surfaces up to
					<select class="field" bind:value={simplify}>
						<option value={0}>nothing (every triangle)</option>
						<option value={0.5}>½ voxel</option>
						<option value={1}>1 voxel</option>
						<option value={2}>2 voxels</option>
					</select>
				</label>
				<button class="btn btn-primary" disabled={!segmentation || active(meshing) || starting}>
					<Box size={13} />
					{active(meshing) ? "Making…" : meshes ? "Make them again" : "Make meshes"}
				</button>
			</form>
			{#if !segmentation && loaded}
				<p class="text-2xs text-ink-dim">Make a final segmentation first.</p>
			{/if}
			{@render progress(meshing, meshError)}
		</div>
	</section>
</div>
