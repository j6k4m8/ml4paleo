<script lang="ts">
	import { goto } from "$app/navigation";
	import { session } from "#lib/session.svelte.ts";
	import Landing from "#lib/ui/Landing.svelte";

	$effect(() => {
		if (session.loaded && session.current) goto(session.current.required_steps.length ? "/account" : "/projects", { replace: true });
	});
</script>

<svelte:head><title>ml4paleo · Machine learning for paleontology</title><meta name="description" content="Open-source, browser-based segmentation of fossil CT and synchrotron X-ray scans. Annotate, train a model, and export 3D meshes." /></svelte:head>

{#if !session.current}<Landing />{/if}
