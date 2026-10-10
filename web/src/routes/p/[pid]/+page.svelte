<script lang="ts">
	import { untrack } from "svelte";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { activityModelHref, groupActivity } from "#lib/pipelines.ts";
	import type { Pipeline, Project, ProjectImage, Upload as UploadInfo } from "#lib/types.ts";
	import { drawImageSlice } from "#lib/thumbnail.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Loop from "#lib/ui/Loop.svelte";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import { uploadFile } from "#lib/upload.ts";
	import { loadLevels } from "#lib/viewer/image.ts";
	import { type Level, PLANES, type Plane } from "#lib/viewer/tiles.ts";
	import Brush from "@lucide/svelte/icons/brush";
	import ExternalLink from "@lucide/svelte/icons/external-link";
	import Upload from "@lucide/svelte/icons/upload";

	const pid = $derived(page.params.pid ?? "");

	let project: Project | null = $state(null);
	let image: ProjectImage | null = $state(null);
	let pipelines: Pipeline[] = $state([]);
	const activity = $derived(groupActivity(pipelines));
	let error = $state("");
	let file: File | null = $state(null);
	let uploading = $state(false);
	let uploaded = $state(0);
	// Replacing a scan is a deliberate step: the drop zone shows only once asked for.
	let replacing = $state(false);
	let levels: Level[] = $state([]);

	// Slices through the middle of the scan, laid out as the annotator's
	// views are: YZ beside XY (sharing y), XZ below it (sharing x).
	const VIEWS: { plane: Plane; label: string; row: number; column: number }[] = [
		{ plane: PLANES.xy, label: "XY", row: 1, column: 1 },
		{ plane: PLANES.yz, label: "YZ", row: 1, column: 2 },
		{ plane: PLANES.xz, label: "XZ", row: 2, column: 1 },
	];

	// Pipelines that bring a scan in: an upload's ingest, or a v1 import.
	const INGESTS = ["ingest", "import"];
	const active = (p: Pipeline) => p.status === "waiting" || p.status === "running";
	// The newest of them (the list is newest first), running or not.
	const latestIngest = $derived(pipelines.find((p) => INGESTS.includes(p.kind)));
	// While one runs the panel shows it instead of a drop zone, so nobody
	// uploads over it thinking the upload failed; and from its finishing until
	// the new image has loaded, so "No image yet" doesn't flash in between.
	let landing: string | null = $state(null);
	const ingesting = $derived(
		latestIngest && (active(latestIngest) || latestIngest.id === landing) ? latestIngest : undefined,
	);
	const failedIngest = $derived(latestIngest?.status === "failed" ? latestIngest : undefined);
	// The file each ingest this page started came from, by pipeline.
	let uploadedNames: Record<string, string> = $state({});
	// Whether the scan is so thin (a single slice, say) that only XY is worth showing.
	const thin = $derived.by(() => {
		if (!image) return false;
		const [, z, y, x] = image.manifest.shape_czyx;
		const [sz, sy, sx] = image.manifest.voxel_size_zyx ?? [1, 1, 1];
		return z * sz < 0.02 * Math.max(x * sx, y * sy);
	});

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
			pipelines = await api<Pipeline[]>(`/api/projects/${pid}/pipelines?include_live_previews=false`);
			const found = await api<ProjectImage>(`/api/projects/${pid}/image`).catch((e: unknown) => {
				if (e instanceof ApiError && e.status === 404) return null;
				throw e;
			});
			// A new object only for a new image, so the previews don't draw again
			// each time something else finishes.
			if (found?.artifact_id !== image?.artifact_id) {
				image = found;
				levels = [];
				// The previews' levels load on their own: if they fail, the page
				// still knows there's a scan (and still asks before replacing it).
				if (found) {
					loadLevels(found.zarr_url).then(
						(loaded) => {
							if (image?.artifact_id === found.artifact_id) levels = loaded;
						},
						() => {},
					);
				}
			}
		} catch (e) {
			error = message(e);
		}
	}

	/** Draw one view of the scan, once its levels are known. */
	function preview(canvas: HTMLCanvasElement, plane: Plane) {
		if (!image || levels.length === 0) return;
		const controller = new AbortController();
		drawImageSlice(canvas, {
			imageUrl: image.zarr_url,
			levels,
			plane,
			window: image.manifest.window,
			target: 256,
			signal: controller.signal,
		}).catch(() => {
			if (!controller.signal.aborted) canvas.title = "No preview";
		});
		return () => controller.abort();
	}

	/** The width / height of a view's plane in physical units, so anisotropic scans keep their shape. */
	function aspect(plane: Plane): string {
		if (!image) return "1";
		const [, z, y, x] = image.manifest.shape_czyx;
		const shape = [z, y, x];
		const size = image.manifest.voxel_size_zyx ?? [1, 1, 1];
		return `${shape[plane.u]! * size[plane.u]!} / ${shape[plane.v]! * size[plane.v]!}`;
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
				if (update.status === "succeeded") {
					if (INGESTS.includes(update.kind)) landing = update.id;
					refresh().finally(() => {
						if (landing === update.id) landing = null;
					});
				}
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
		if (image && !confirm(`Replace the scan with ${file.name}? Predictions and results made from the current scan won't match the new one.`)) {
			return;
		}
		uploading = true;
		error = "";
		try {
			const existing = await unfinished(file);
			const done = await uploadFile(pid, file, (fraction) => (uploaded = fraction), existing);
			const pipeline = await api<Pipeline>(`/api/projects/${pid}/ingest`, {
				body: { upload_id: done.id },
			});
			uploadedNames = { ...uploadedNames, [pipeline.id]: file.name };
			file = null;
			replacing = false;
			await refresh();
		} catch (e) {
			// Shown by the drop zone, where the person is looking.
			rejected = message(e);
		} finally {
			uploading = false;
		}
	}

	async function stop(pipeline: Pipeline) {
		if (!confirm("Stop bringing this scan in?")) return;
		try {
			await api(`/api/projects/${pid}/pipelines/${pipeline.id}/cancel`, { method: "POST" });
		} catch (e) {
			error = message(e);
		}
		await refresh();
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

{#snippet bringingIn(pipeline: Pipeline)}
	{@const name = uploadedNames[pipeline.id]}
	<div class="flex flex-col gap-2 rounded-sm border border-edge bg-field p-3" role="status">
		{#if active(pipeline)}
			<span class="[overflow-wrap:anywhere]">
				{pipeline.kind === "import" ? "Importing a v1 job." : name ? `Uploaded ${name}.` : "A scan was uploaded."} Bringing it
				in: reading its slices and building the zoom levels.
			</span>
			<div class="flex items-center gap-2">
				<progress class="h-1.5 flex-1" max="1" value={pipeline.progress}></progress>
				<span class="font-mono text-2xs text-ink-dim">{percent(pipeline.progress)}</span>
				<button class="btn btn-ghost" onclick={() => stop(pipeline)}>Stop</button>
			</div>
			<span class="text-2xs text-ink-dim">You can leave this page; it keeps going.</span>
		{:else}
			<span>Brought in. Loading it…</span>
		{/if}
	</div>
{/snippet}

{#snippet failed()}
	{#if failedIngest}
		<p class="error [overflow-wrap:anywhere]" role="alert">
			The last scan couldn't be brought in{failedIngest.error ? `: ${failedIngest.error}` : "."}
		</p>
	{/if}
{/snippet}

{#snippet dropzone(prompt: string)}
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
			<span>{file ? file.name : prompt}</span>
			<span class="text-2xs text-ink-faint">A .zip of image slices or DICOM files</span>
			<input class="sr-only" type="file" accept=".zip" onchange={(e) => choose(e.currentTarget.files?.[0])} disabled={uploading} />
		</label>
		{#if rejected}<p class="error" role="alert">{rejected}</p>{/if}
		{#if uploading}
			<div class="flex items-center gap-2">
				<progress class="h-1.5 flex-1" max="1" value={uploaded}></progress>
				<span class="font-mono text-2xs text-ink-dim">{percent(uploaded)}</span>
			</div>
		{/if}
		<div class="flex items-center gap-2">
			<button class="btn btn-primary" disabled={!file || uploading}>Upload and ingest</button>
			{#if image && !uploading}
				<button
					type="button"
					class="btn btn-ghost"
					onclick={() => {
						replacing = false;
						file = null;
						rejected = "";
					}}>Cancel</button
				>
			{/if}
		</div>
	</form>
{/snippet}

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
						{@const [sz, , sx] = image.manifest.voxel_size_zyx ?? [1, 1, 1]}
						{@const widest = Math.max(x * sx, z * sz)}
						<a
							href="/p/{pid}/annotate"
							class="grid max-w-xl gap-1 hover:no-underline"
							style:grid-template-columns={thin ? "1fr" : `${(x * sx) / widest}fr minmax(1.5rem, ${(z * sz) / widest}fr)`}
							title="Open in the annotator"
						>
							{#each thin ? VIEWS.slice(0, 1) : VIEWS as view (view.label)}
								<div class="relative min-w-0" style:grid-row={view.row} style:grid-column={view.column}>
									<canvas
										class="block w-full rounded-sm border border-edge bg-pasteboard"
										style:aspect-ratio={aspect(view.plane)}
										{@attach (canvas) => preview(canvas, view.plane)}
									></canvas>
									<span class="pointer-events-none absolute top-1 left-1.5 text-2xs text-ink-faint">{view.label}</span>
								</div>
							{/each}
						</a>
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
						{#if image.manifest.skipped_count}
							<p class="text-2xs text-ink-dim [overflow-wrap:anywhere]">
								Left out {image.manifest.skipped_count === 1 ? "a file" : `${image.manifest.skipped_count} files`} that
								{image.manifest.skipped_count === 1 ? "isn't a slice" : "aren't slices"}: {image.manifest.skipped?.join(", ")}{(image.manifest
									.skipped?.length ?? 0) < image.manifest.skipped_count
									? ", …"
									: ""}
							</p>
						{/if}
						{#each image.manifest.notes ?? [] as note, index (index)}
							<p class="text-2xs text-ink-dim [overflow-wrap:anywhere]">{note}</p>
						{/each}
						<div class="flex items-center gap-3">
							<a class="flex items-center gap-1" href={image.neuroglancer_url} target="_blank" rel="noopener">
								Open in Neuroglancer <ExternalLink size={12} />
							</a>
							{#if !replacing && !ingesting && !uploading}
								<button class="btn btn-ghost ml-auto" onclick={() => (replacing = true)}>Replace the scan…</button>
							{/if}
						</div>
						{#if ingesting && !uploading}
							{@render bringingIn(ingesting)}
						{:else if replacing || uploading}
							<div class="flex flex-col gap-2 border-t border-edge pt-3">
								{@render failed()}
								<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">
									A new scan replaces this one for everyone in the project. Predictions and results made from this one won't
									match it; labels stay where they are, so they only line up if the new scan covers the same space.
								</p>
								{@render dropzone("Drop the new scan here, or click to choose")}
							</div>
						{/if}
					{:else if ingesting && !uploading}
						{@render bringingIn(ingesting)}
					{:else}
						<p class="muted">No image yet. Upload a zip of image slices or DICOM files.</p>
						{@render failed()}
						{@render dropzone("Drop a scan here, or click to choose")}
					{/if}
				</div>
			</section>

			{#if image}
				<Loop {pid} importHref="/p/{pid}/labels" />
			{/if}
		</div>

		<section class="panel self-start">
			<h2 class="panel-title">Activity</h2>
			{#if pipelines.length === 0}
				<p class="muted p-3">Nothing has run yet.</p>
			{:else}
				<ul class="divide-y divide-edge">
					{#each activity as group (group.latest.id)}
						{@const pipeline = group.latest}
						{@const modelHref = activityModelHref(pid, group)}
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
								<span class="shrink-0 font-medium capitalize">{pipeline.kind}{#if group.count > 1} (×{group.count}){/if}</span>
								<time class="ml-auto text-right text-2xs text-ink-faint" datetime={pipeline.created_at} title={group.count > 1 ? "Latest run" : undefined}>{new Date(pipeline.created_at).toLocaleString()}</time>
							</div>
							{#if pipeline.status === "waiting" || pipeline.status === "running"}
								<div class="flex items-center gap-2">
									<progress class="h-1 flex-1" max="1" value={pipeline.progress}></progress>
									<span class="font-mono text-2xs text-ink-dim">{percent(pipeline.progress)}</span>
								</div>
							{:else}
								<div class="flex items-center justify-between gap-2 text-2xs">
									<span class="text-ink-dim">{pipeline.status}</span>
									{#if modelHref}<a href={modelHref}>{group.count > 1 ? "Latest model" : "View model"}</a>{/if}
								</div>
							{/if}
							{#if active(pipeline) && modelHref}<a class="text-2xs" href={modelHref}>View model</a>{/if}
							{#if pipeline.error}<span class="text-2xs text-danger">{pipeline.error}</span>{/if}
						</li>
					{/each}
				</ul>
			{/if}
		</section>
	</div>
{/if}
{#if error}<p class="error p-6" role="alert">{error}</p>{/if}
