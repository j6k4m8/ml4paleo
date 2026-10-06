<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import type { ProjectImage } from "#lib/types.ts";
	import Viewer from "#lib/viewer/Viewer.svelte";

	const pid = $derived(page.params.pid ?? "");
	let image: ProjectImage | null = $state(null);
	let error = $state("");

	$effect(() => {
		api<ProjectImage>(`/api/projects/${pid}/image`).then(
			(found) => (image = found),
			(e) => (error = message(e)),
		);
	});
</script>

<div class="page">
	<p><a href="/p/{pid}">← Project</a></p>
	{#if image}
		{#key image.artifact_id}
			<Viewer {image} projectId={pid} roi={page.url.searchParams.get("roi")} />
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
