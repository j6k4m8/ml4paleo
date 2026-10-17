<script lang="ts">
	import type { Snippet } from "svelte";
	import { withBackground } from "#lib/viewer/background.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";

	let { classes, counts, actions }: { classes: LabelClass[]; counts: Record<string, number>; actions?: Snippet<[LabelClass]> } = $props();
</script>

<!-- What has been labeled: background first, then each class, with its voxels. -->
<ul class="flex flex-col gap-0.5">
	{#each withBackground(classes) as c (c.value)}
		<li class="flex items-center gap-2">
			<span class="size-2.5 shrink-0 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={c.color}></span>
			<span class="flex-1 truncate">{c.name}</span>
			<span class="font-mono text-2xs text-ink-dim">{(counts[c.value] ?? 0).toLocaleString()} voxels</span>
			{#if actions}{@render actions(c)}{/if}
		</li>
	{/each}
</ul>
