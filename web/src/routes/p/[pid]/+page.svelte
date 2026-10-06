<script lang="ts">
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import type { Pipeline, Project, ProjectImage } from "#lib/types.ts";
	import { uploadFile } from "#lib/upload.ts";

	const pid = $derived(page.params.pid ?? "");

	let project: Project | null = $state(null);
	let image: ProjectImage | null = $state(null);
	let pipelines: Pipeline[] = $state([]);
	let error = $state("");
	let file: File | null = $state(null);
	let uploading = $state(false);
	let uploaded = $state(0);

	const running = $derived(pipelines.filter((p) => p.status === "waiting" || p.status === "running"));

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

	// Follow each running pipeline until it finishes.
	$effect(() => {
		const sources = running.map((pipeline) => {
			const source = new EventSource(`/api/projects/${pid}/pipelines/${pipeline.id}/events`);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				pipelines = pipelines.map((p) => (p.id === update.id ? update : p));
				if (update.status === "succeeded") refresh();
			});
			source.onerror = () => source.close();
			return source;
		});
		return () => sources.forEach((source) => source.close());
	});

	async function upload(event: SubmitEvent) {
		event.preventDefault();
		if (!file) return;
		uploading = true;
		error = "";
		try {
			const done = await uploadFile(pid, file, (fraction) => (uploaded = fraction));
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

{#if project}
	<h1>{project.name}</h1>

	<section>
		<h2>Image</h2>
		{#if image}
			{@const [, z, y, x] = image.manifest.shape_czyx}
			<p>{x} × {y} × {z} voxels, {image.manifest.dtype}</p>
			<p>
				<a href="/p/{pid}/annotate">Open the annotator</a>
				{#if image.neuroglancer_url}
					· <a href={image.neuroglancer_url} target="_blank" rel="noopener">Open in Neuroglancer</a>
				{/if}
			</p>
		{:else}
			<p class="muted">No image yet. Upload a zip of image slices or DICOM files.</p>
		{/if}
		<form class="stack" onsubmit={upload}>
			<label>
				{image ? "Replace the image" : "Image archive"}
				<input
					type="file"
					accept=".zip"
					onchange={(e) => (file = e.currentTarget.files?.[0] ?? null)}
					disabled={uploading}
				/>
			</label>
			{#if uploading}
				<progress max="1" value={uploaded}></progress>
				<span class="muted">Uploading {percent(uploaded)}</span>
			{/if}
			<button disabled={!file || uploading}>Upload</button>
		</form>
	</section>

	{#if pipelines.length > 0}
		<section>
			<h2>Pipelines</h2>
			<table>
				<thead><tr><th>Kind</th><th>Status</th><th>Progress</th><th>Started</th></tr></thead>
				<tbody>
					{#each pipelines as pipeline (pipeline.id)}
						<tr>
							<td>{pipeline.kind}</td>
							<td>
								{pipeline.status}
								{#if pipeline.error}<span class="error">: {pipeline.error}</span>{/if}
							</td>
							<td><progress max="1" value={pipeline.progress}></progress> {percent(pipeline.progress)}</td>
							<td>{new Date(pipeline.created_at).toLocaleString()}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}
{/if}
{#if error}<p class="error" role="alert">{error}</p>{/if}

<style>
	table {
		border-collapse: collapse;
	}
	th,
	td {
		text-align: left;
		padding: 0.25rem 0.75rem 0.25rem 0;
		border-bottom: 1px solid var(--line);
	}
</style>
