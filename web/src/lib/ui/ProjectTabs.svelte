<script lang="ts">
	import Brain from "@lucide/svelte/icons/brain";
	import Brush from "@lucide/svelte/icons/brush";
	import Shapes from "@lucide/svelte/icons/shapes";
	import House from "@lucide/svelte/icons/house";
	import Settings from "@lucide/svelte/icons/settings";
	import SquareDashed from "@lucide/svelte/icons/square-dashed";
	import { page } from "$app/state";

	let { pid }: { pid: string } = $props();

	const tabs = $derived([
		{ href: `/p/${pid}`, label: "Overview", icon: House },
		{ href: `/p/${pid}/annotate`, label: "Annotate", icon: Brush },
		{ href: `/p/${pid}/rois`, label: "ROIs", icon: SquareDashed },
		{ href: `/p/${pid}/models`, label: "Models", icon: Brain },
		{ href: `/p/${pid}/results`, label: "Results", icon: Shapes },
		{ href: `/p/${pid}/settings`, label: "Settings", icon: Settings },
	]);
</script>

<!-- A project's pages as document-style tabs. -->
<nav class="flex h-8 items-end gap-px border-b border-edge bg-chrome px-3" aria-label="Project">
	{#each tabs as tab (tab.href)}
		{@const current = page.url.pathname === tab.href}
		<a
			href={tab.href}
			aria-current={current ? "page" : undefined}
			class="flex h-7 items-center gap-1.5 rounded-t-sm px-3 text-xs no-underline hover:no-underline
				{current ? 'bg-pasteboard text-ink shadow-[inset_0_1px_0_var(--color-accent)]' : 'text-ink-dim hover:bg-panel hover:text-ink'}"
		>
			<tab.icon size={13} />
			{tab.label}
		</a>
	{/each}
</nav>
