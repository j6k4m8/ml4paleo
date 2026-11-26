<script lang="ts">
	import Compass from "@lucide/svelte/icons/compass";
	import { goto } from "$app/navigation";
	import { message } from "#lib/api.ts";
	import { SHOW_ROIS } from "#lib/features.ts";
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
	<h2 class="panel-title">How to segment a scan</h2>
	<ol class="flex flex-col gap-2 p-3">
		<li class="flex gap-2">
			{@render step(1, "Label")}
			<span class="text-ink-dim">
				{#if SHOW_ROIS}
					a few places: draw an ROI in the <a href="/p/{pid}/annotate">annotator</a>, paint it, and mark it complete. Or
					<a href={importHref}>import labels</a> from a file.
				{:else}
					Open <a href="/p/{pid}/annotate">Annotate</a>, choose Add label for what you want to find, then use the Brush to
					paint examples of that label and Background. You can also <a href={importHref}>import labels</a> from a file.
				{/if}
			</span>
		</li>
		<li class="flex gap-2">
			{@render step(2, "Train")}
			<span class="text-ink-dim">On <a href="/p/{pid}/models">Models</a>, choose Train model.</span>
		</li>
		<li class="flex gap-2">
			{#if SHOW_ROIS}
				{@render step(3, "Explore")}
				<span class="text-ink-dim">somewhere new to see how the model does, and fix what it gets wrong.</span>
			{:else}
				{@render step(3, "Predict")}
				<span class="text-ink-dim">
					When training finishes, choose Predict full scan on Models. Turn on the prediction layer in
					<a href="/p/{pid}/neuroglancer">Visualization</a> to check what the model found.
				</span>
			{/if}
		</li>
		<li class="flex gap-2">
			{#if SHOW_ROIS}
				{@render step(4, "Retrain")}
				<span class="text-ink-dim">with those fixes, then explore again.</span>
			{:else}
				{@render step(4, "Refine")}
				<span class="text-ink-dim">
					Add more examples in <a href="/p/{pid}/annotate">Annotate</a> where the prediction needs work, then train and predict again.
				</span>
			{/if}
		</li>
	</ol>
	{#if SHOW_ROIS}
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
	{/if}
</section>
