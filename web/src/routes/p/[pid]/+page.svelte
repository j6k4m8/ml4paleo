<script lang="ts">
	import { untrack } from "svelte";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import type { Pipeline, Project, ProjectImage, Upload as UploadInfo } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Loop from "#lib/ui/Loop.svelte";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import { uploadFile } from "#lib/upload.ts";
	import Brush from "@lucide/svelte/icons/brush";
	import ExternalLink from "@lucide/svelte/icons/external-link";
	import Upload from "@lucide/svelte/icons/upload";

	const pid = $derived(page.params.pid ?? "");

	let project: Project | null = $state(null);
	let image: ProjectImage | null = $state(null);
	let pipelines: Pipeline[] = $state([]);
	let error = $state("");
	let file: File | null = $state(null);
	let uploading = $state(false);
	let uploaded = $state(0);

	// Only which pipelines run, so status updates don't reopen the streams.
	const running = $derived(
		pipelines
			.filter((p) => p.status === "waiting" || p.status === "running")
			.map((p) => p.id)
			.join(","),
	);

	let dragging = $state(false);
	let rejected = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: project?.name ?? "…" }]);
	});

	/** Take a picked or dropped file, if it's a zip. */
	function choose(chosen: File | undefined) {
		if (!chosen || uploading) return;
		if (!chosen.name.toLowerCase().endsWith(".zip")) {
			rejected = `${chosen.name} isn't a .zip. Choose a .zip of image slices or DICOM files.`;
			return;
		}
		rejected = "";
		file = chosen;
	}

	function dropped(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		choose(event.dataTransfer?.files?.[0]);
	}

	async function refresh() {
		try {
			project = await api<Project>(`/api/projects/${pid}`);
			pipelines = await api<Pipeline[]>(`/api/projects/${pid}/pipelines`);
			image = await api<ProjectImage>(`/api/projects/${pid}/image`).catch((e: unknown) => {
				if (e instanceof ApiError && e.status === 404) return null;
				throw e;
			});
		} catch (e) {
			error = message(e);
		}
	}

	$effect(() => {
		if (pid) refresh();
	});

	// Follow each running pipeline until it finishes. The browser reconnects
	// dropped streams; the server ends them (204) once the pipeline is done.
	$effect(() => {
		const ids = running ? running.split(",") : [];
		const sources = ids.map((id) => {
			const source = new EventSource(`/api/projects/${untrack(() => pid)}/pipelines/${id}/events`);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				pipelines = untrack(() => pipelines).map((p) => (p.id === update.id ? update : p));
				if (update.status === "succeeded") refresh();
			});
			return source;
		});
		return () => sources.forEach((source) => source.close());
	});

	/** An unfinished upload of this same file, to resume. */
	async function unfinished(chosen: File): Promise<UploadInfo | undefined> {
		const uploads = await api<UploadInfo[]>(`/api/projects/${pid}/uploads`);
		return uploads.find((u) => u.state === "uploading" && u.filename === chosen.name && u.size === chosen.size);
	}

	async function upload(event: SubmitEvent) {
		event.preventDefault();
		if (!file) return;
		uploading = true;
		error = "";
		try {
			const existing = await unfinished(file);
			const done = await uploadFile(pid, file, (fraction) => (uploaded = fraction), existing);
			const pipeline = await api<Pipeline>(`/api/projects/${pid}/ingest`, {
				body: { upload_id: done.id },
			});
			pipelines = [pipeline, ...pipelines];
			file = null;
		} catch (e) {
			error = message(e);
		} finally {
			uploading = false;
		}
	}

	function percent(fraction: number): string {
		return `${Math.round(fraction * 100)}%`;
	}
</script>

<!-- A file dropped just outside the drop zone shouldn't open in the tab. -->
<svelte:window
	ondragover={(e) => {
		if (e.defaultPrevented || !e.dataTransfer) return;
		e.preventDefault();
		e.dataTransfer.dropEffect = "none";
	}}
	ondrop={(e) => e.preventDefault()}
/>

<ProjectTabs {pid} />

{#if project}
	<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-[1fr_22rem]">
		<div class="flex flex-col gap-4">
			<div class="flex items-center gap-3">
				<h1 class="text-lg">{project.name}</h1>
				{#if image}
					<a class="btn btn-primary ml-auto h-7 hover:no-underline" href="/p/{pid}/annotate">
						<Brush size={14} /> Open in annotator
					</a>
				{/if}
			</div>

			<section class="panel">
				<h2 class="panel-title">Image</h2>
				<div class="flex flex-col gap-3 p-3">
					{#if image}
						{@const [, z, y, x] = image.manifest.shape_czyx}
						<dl class="grid grid-cols-[8rem_1fr] gap-y-1">
							<dt class="text-ink-dim">Size</dt>
							<dd class="font-mono">{x} × {y} × {z} voxels</dd>
							<dt class="text-ink-dim">Type</dt>
							<dd class="font-mono">{image.manifest.dtype}</dd>
							{#if image.manifest.voxel_size_zyx}
								<dt class="text-ink-dim">Voxel size</dt>
								<dd class="font-mono">
									{image.manifest.voxel_size_zyx.slice().reverse().join(" × ")}
									{image.manifest.unit ?? ""}
								</dd>
							{/if}
						</dl>
						{#if image.neuroglancer_url}
							<a class="flex items-center gap-1 self-start" href={image.neuroglancer_url} target="_blank" rel="noopener">
								Open in Neuroglancer <ExternalLink size={12} />
							</a>
						{/if}
					{:else}
						<p class="muted">No image yet. Upload a zip of image slices or DICOM files.</p>
					{/if}

					<form onsubmit={upload} class="flex flex-col gap-2">
						<!-- Its children ignore the pointer, so dragging over them doesn't count as leaving. -->
						<label
							class="flex cursor-pointer flex-col items-center gap-1.5 rounded-sm border border-dashed p-6 text-center transition-colors *:pointer-events-none
								has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-accent
								{dragging ? 'border-accent bg-accent-soft/40' : 'border-line bg-field hover:border-ink-faint'}"
							ondragover={(e) => {
								e.preventDefault();
								dragging = true;
							}}
							ondragleave={() => (dragging = false)}
							ondrop={dropped}
						>
							<Upload size={20} class="text-ink-faint" />
							<span>{file ? file.name : image ? "Drop a new scan here to replace the image" : "Drop a scan here, or click to choose"}</span>
							<span class="text-2xs text-ink-faint">A .zip of image slices or DICOM files</span>
							<input
								class="sr-only"
								type="file"
								accept=".zip"
								onchange={(e) => choose(e.currentTarget.files?.[0])}
								disabled={uploading}
							/>
						</label>
						{#if rejected}<p class="error" role="alert">{rejected}</p>{/if}
						{#if uploading}
							<div class="flex items-center gap-2">
								<progress class="h-1.5 flex-1" max="1" value={uploaded}></progress>
								<span class="font-mono text-2xs text-ink-dim">{percent(uploaded)}</span>
							</div>
						{/if}
						<button class="btn btn-primary self-start" disabled={!file || uploading}>Upload and ingest</button>
					</form>
				</div>
			</section>

			{#if image}
				<Loop {pid} importHref="/p/{pid}/rois#import-labels" />
			{/if}
		</div>

		<section class="panel self-start">
			<h2 class="panel-title">Activity</h2>
			{#if pipelines.length === 0}
				<p class="muted p-3">Nothing has run yet.</p>
			{:else}
				<ul class="divide-y divide-edge">
					{#each pipelines as pipeline (pipeline.id)}
						<li class="flex flex-col gap-1 px-3 py-2">
							<div class="flex items-center gap-2">
								<span
									class="size-1.5 rounded-full {pipeline.status === 'succeeded'
										? 'bg-ok'
										: pipeline.status === 'failed'
											? 'bg-danger'
											: pipeline.status === 'cancelled'
												? 'bg-ink-faint'
												: 'animate-pulse bg-warn'}"
								></span>
								<span class="font-medium capitalize">{pipeline.kind}</span>
								<span class="ml-auto text-2xs text-ink-faint">{new Date(pipeline.created_at).toLocaleString()}</span>
							</div>
							{#if pipeline.status === "waiting" || pipeline.status === "running"}
								<div class="flex items-center gap-2">
									<progress class="h-1 flex-1" max="1" value={pipeline.progress}></progress>
									<span class="font-mono text-2xs text-ink-dim">{percent(pipeline.progress)}</span>
								</div>
							{:else}
								<span class="text-2xs text-ink-dim">{pipeline.status}</span>
							{/if}
							{#if pipeline.error}<span class="text-2xs text-danger">{pipeline.error}</span>{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	</div>
{/if}
{#if error}<p class="error p-6" role="alert">{error}</p>{/if}
