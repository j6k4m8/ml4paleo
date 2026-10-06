<script lang="ts">
	import ChevronDown from "@lucide/svelte/icons/chevron-down";
	import ChevronRight from "@lucide/svelte/icons/chevron-right";
	import type { Snippet } from "svelte";

	let {
		title,
		open = $bindable(true),
		actions,
		children,
		class: extra = "",
	}: {
		title: string;
		open?: boolean;
		actions?: Snippet;
		children: Snippet;
		class?: string;
	} = $props();
</script>

<!-- A docked panel: a slim header that folds it away, and its contents. -->
<section class="border-b border-edge bg-panel {extra}">
	<header class="flex h-7 items-center gap-1 bg-panel-header pr-1.5 pl-1">
		<button
			class="flex h-full flex-1 items-center gap-1 text-left text-2xs font-semibold tracking-wide text-ink-dim uppercase hover:text-ink"
			aria-expanded={open}
			onclick={() => (open = !open)}
		>
			{#if open}<ChevronDown size={12} />{:else}<ChevronRight size={12} />{/if}
			{title}
		</button>
		{@render actions?.()}
	</header>
	{#if open}
		<div class="flex flex-col gap-2 p-2.5">
			{@render children()}
		</div>
	{/if}
</section>
