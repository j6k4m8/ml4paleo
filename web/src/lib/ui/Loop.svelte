<script lang="ts">
	import Compass from "@lucide/svelte/icons/compass";
	import { goto } from "$app/navigation";
	import { message } from "#lib/api.ts";
	import { explore } from "#lib/rois.svelte.ts";

	let {
		pid,
		importHref,
		hasImage = true,
	}: {
		pid: string;
		importHref: string;
		/** Whether the project has an image to explore; null until known. */
		hasImage?: boolean | null;
	} = $props();

	let exploring = $state(false);
	let error = $state("");

	/** Make an ROI somewhere new, start a proposal there if a model is ready, and open it. */
	async function go() {
		exploring = true;
		error = "";
		try {
			const roi = await explore(pid);
			await goto(`/p/${pid}/annotate?roi=${roi.id}`);
		} catch (e) {
			error = message(e);
		} finally {
			exploring = false;
		}
	}
</script>

{#snippet step(number: number, name: string)}
	<span class="grid size-4 shrink-0 place-items-center rounded-full bg-raised font-mono text-2xs text-ink">{number}</span>
	<span class="font-medium">{name}</span>
{/snippet}

<!-- How labels become a model, and how a model gets better: the loop. -->
<section class="panel">
	<h2 class="panel-title">How training works</h2>
	<ol class="flex flex-col gap-2 p-3">
		<li class="flex gap-2">
			{@render step(1, "Label")}
			<span class="text-ink-dim">
				a few places: draw an ROI in the <a href="/p/{pid}/annotate">annotator</a>, paint it, and mark it complete. Or
				<a href={importHref}>import labels</a> from a file.
			</span>
		</li>
		<li class="flex gap-2">
			{@render step(2, "Train")}
			<span class="text-ink-dim">a model on the <a href="/p/{pid}/models">Models</a> page.</span>
		</li>
		<li class="flex gap-2">
			{@render step(3, "Explore")}
			<span class="text-ink-dim">somewhere new to see how the model does, and fix what it gets wrong.</span>
		</li>
		<li class="flex gap-2">
			{@render step(4, "Retrain")}
			<span class="text-ink-dim">with those fixes, then explore again.</span>
		</li>
	</ol>
	<div class="flex flex-col gap-1.5 border-t border-edge p-3">
		<button
			class="btn btn-primary self-start"
			onclick={go}
			disabled={exploring || !hasImage}
			title={hasImage === false ? "Ingest a scan first" : undefined}
		>
			<Compass size={13} />
			{exploring ? "Finding a place…" : "Explore a new place"}
		</button>
		<p class="text-2xs text-ink-faint">
			Opens a random spot no ROI covers. If a model is ready, it proposes labels there for you to check.
		</p>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
	</div>
</section>
