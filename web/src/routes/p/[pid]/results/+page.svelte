<script lang="ts">
	import Archive from "@lucide/svelte/icons/archive";
	import Box from "@lucide/svelte/icons/box";
	import Brush from "@lucide/svelte/icons/brush";
	import Combine from "@lucide/svelte/icons/combine";
	import Download from "@lucide/svelte/icons/download";
	import Shapes from "@lucide/svelte/icons/shapes";
	import Trash from "@lucide/svelte/icons/trash";
	import { untrack } from "svelte";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import type { Pipeline } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";

	interface Prediction {
		artifact_id: string;
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
		artifact_id: string;
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

	type Source = "image" | "prediction" | "segmentation" | "meshes";

	interface Export {
		id: string;
		source: Source;
		source_artifact_id: string | null;
		format: string;
		filename: string;
		status: "making" | "ready" | "failed";
		pipeline_id: string | null;
		progress: number;
		error: string | null;
		bytes: number;
		created_at: string;
		expires_at: string;
		download_url: string | null;
	}

	const SLICES: [string, string][] = [
		["tiff", "TIFF stack"],
		["png", "PNG stack"],
	];
	const SOURCES: { source: Source; label: string; formats: [string, string][] }[] = [
		{ source: "image", label: "Image", formats: [["zarr", "OME-Zarr"], ...SLICES] },
		{ source: "prediction", label: "Prediction", formats: [["zarr", "Zarr"], ...SLICES] },
		{ source: "segmentation", label: "Final segmentation", formats: [["zarr", "Zarr"], ...SLICES] },
		{ source: "meshes", label: "Meshes", formats: [["zip", "Every file"]] },
	];

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
	// The image's dtype (numpy's notation, such as "<u2"), once there is one.
	let imageDtype: string | null = $state(null);
	let imageId: string | null = $state(null);
	let exports: Export[] = $state([]);
	let pipelines: Pipeline[] = $state([]);
	let loaded = $state(false);
	let minVoxels = $state(50);
	let downsample = $state(1);
	let method = $state("any");
	let simplify = $state(1);
	let error = $state("");
	let composeError = $state("");
	let meshError = $state("");
	let exportError = $state("");
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
			let image: { artifact_id: string; manifest: { dtype: string } } | null;
			[prediction, segmentation, meshes, pipelines, image, exports] = await Promise.all([
				api<Prediction>(`/api/projects/${pid}/prediction`).catch(missing),
				api<Segmentation>(`/api/projects/${pid}/segmentation`).catch(missing),
				api<Meshes>(`/api/projects/${pid}/meshes`).catch(missing),
				api<Pipeline[]>(`/api/projects/${pid}/pipelines`),
				api<{ artifact_id: string; manifest: { dtype: string } }>(`/api/projects/${pid}/image`).catch(missing),
				api<Export[]>(`/api/projects/${pid}/exports`),
			]);
			imageDtype = image?.manifest.dtype ?? null;
			imageId = image?.artifact_id ?? null;
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
		[
			...[composing, meshing].filter(active).map((p) => p?.id),
			...exports.filter((e) => e.status === "making").map((e) => e.pipeline_id),
		]
			.filter(Boolean)
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
				exports = untrack(() => exports).map((e) => (e.pipeline_id === update.id ? { ...e, progress: update.progress } : e));
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

	const available = (source: Source) =>
		({ image: imageDtype !== null, prediction: !!prediction, segmentation: !!segmentation, meshes: !!meshes })[source];
	// PNG holds 8- or 16-bit unsigned values; labels are 8-bit.
	const pngHolds = (source: Source) => source !== "image" || /u[12]$/.test(imageDtype ?? "");

	function exportAs(source: Source, format: string) {
		start(`/api/projects/${pid}/exports`, { source, format }, (text) => (exportError = text));
	}

	// What each source is now, to flag exports made from an older one.
	const current = $derived.by(
		(): Record<Source, string | null> => ({
			image: imageId,
			prediction: prediction?.artifact_id ?? null,
			segmentation: segmentation?.artifact_id ?? null,
			meshes: meshes?.artifact_id ?? null,
		}),
	);
	let forgetting = $state(new Set<string>());

	async function forget(item: Export) {
		if (forgetting.has(item.id)) return;
		forgetting = new Set([...forgetting, item.id]);
		exportError = "";
		try {
			await api(`/api/projects/${pid}/exports/${item.id}`, { method: "DELETE" });
		} catch (e) {
			exportError = message(e);
		}
		await refresh();
		forgetting = new Set([...forgetting].filter((id) => id !== item.id));
	}

	function size(bytes: number): string {
		if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
		if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
		return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
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

	<section class="panel self-start lg:col-span-2">
		<h2 class="panel-title">Downloads</h2>
		<div class="flex flex-col gap-3 p-3">
			<p class="text-ink-dim">
				Zip archives of the project's volumes and meshes. Stacks hold one image per slice, as the scan was uploaded. Archives
				are kept for a week, and count against your storage until then.
			</p>
			<div class="grid items-center gap-2 sm:grid-cols-[max-content_1fr]">
				{#each SOURCES as { source, label, formats } (source)}
					<span class="font-medium" class:text-ink-faint={!available(source)}>{label}</span>
					<div class="flex flex-wrap gap-1.5">
						{#each formats as [format, name] (format)}
							{@const holds = format !== "png" || pngHolds(source)}
							<button
								class="btn"
								disabled={!available(source) || !holds || starting}
								title={holds ? undefined : "PNG holds 8- or 16-bit unsigned values; use TIFF for this image"}
								onclick={() => exportAs(source, format)}
							>
								<Archive size={13} />
								{name}
							</button>
						{/each}
					</div>
				{/each}
			</div>
			{#if exports.length}
				<ul class="flex flex-col divide-y divide-edge rounded-sm border border-edge bg-field">
					{#each exports as item (item.id)}
						<li class="flex flex-wrap items-center gap-2 px-2 py-1.5">
							<span class="flex min-w-0 flex-1 flex-col">
								<span class="truncate font-mono text-2xs">{item.filename}</span>
								<span class="text-2xs text-ink-dim">
									{new Date(item.created_at).toLocaleString()}
									{#if item.source_artifact_id && current[item.source] && item.source_artifact_id !== current[item.source]}
										· <span class="text-warn">from an older {item.source}</span>
									{/if}
								</span>
							</span>
							{#if item.status === "making"}
								<progress class="h-1 w-32 accent-accent" max="1" value={item.progress}></progress>
								<span class="font-mono text-2xs text-ink-dim">{Math.round(item.progress * 100)}%</span>
							{:else if item.status === "ready" && item.download_url}
								<span class="text-2xs text-ink-dim">
									{size(item.bytes)} · kept until {new Date(item.expires_at).toLocaleString()}
								</span>
								<a class="btn btn-primary hover:no-underline" href={item.download_url} download={item.filename}>
									<Download size={13} />
									Download
								</a>
							{:else}
								<span class="error text-2xs" role="alert">{item.error}</span>
							{/if}
							<button
								class="btn btn-ghost"
								disabled={forgetting.has(item.id)}
								title={item.status === "making" ? "Stop" : "Delete"}
								aria-label="{item.status === 'making' ? 'Stop' : 'Delete'} {item.filename}"
								onclick={() => forget(item)}
							>
								<Trash size={13} />
							</button>
						</li>
					{/each}
				</ul>
			{/if}
			{#if exportError}<p class="error" role="alert">{exportError}</p>{/if}
		</div>
	</section>
</div>
