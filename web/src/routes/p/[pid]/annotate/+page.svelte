<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import type { ProjectImage } from "#lib/types.ts";
	import Viewer from "#lib/viewer/Viewer.svelte";

	const pid = $derived(page.params.pid ?? "");
	// The image and the project it belongs to, set together.
	let shown: { pid: string; image: ProjectImage } | null = $state(null);
	let error = $state("");

	$effect(() => {
		const project = pid;
		const controller = new AbortController();
		shown = null;
		error = "";
		api<ProjectImage>(`/api/projects/${project}/image`, { signal: controller.signal }).then(
			(image) => (shown = { pid: project, image }),
			(e) => {
				if (!controller.signal.aborted) error = message(e);
			},
		);
		return () => controller.abort();
	});
</script>

<div class="page">
	<p><a href="/p/{pid}">← Project</a></p>
	{#if shown}
		{#key `${shown.pid}/${shown.image.artifact_id}`}
			<Viewer image={shown.image} projectId={shown.pid} roi={page.url.searchParams.get("roi")} />
		{/key}
	{:else if error}
		<p class="error" role="alert">{error}</p>
	{/if}
</div>

<style>
	.page {
		display: flex;
		flex-direction: column;
		height: 100%;
	}
	.page p {
		margin: 0 0 0.5rem;
	}
</style>
