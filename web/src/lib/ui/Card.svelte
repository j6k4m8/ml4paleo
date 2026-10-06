<script lang="ts">
	import type { Snippet } from "svelte";

	let { title, children, footer }: { title: string; children: Snippet; footer?: Snippet } = $props();
</script>

<!-- A small dialog-like card centered on the pasteboard, for sign-in and account pages. -->
<div class="grid min-h-full place-items-center p-6">
	<div class="panel w-full max-w-sm shadow-xl shadow-black/30">
		<h1 class="panel-title !text-xs !normal-case">{title}</h1>
		<div class="flex flex-col gap-3 p-4">
			{@render children()}
		</div>
		{#if footer}
			<div class="border-t border-edge px-4 py-2.5 text-ink-dim">{@render footer()}</div>
		{/if}
	</div>
</div>
