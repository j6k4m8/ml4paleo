<script lang="ts">
	import type { Component } from "svelte";

	let {
		icon: Icon,
		label,
		shortcut = "",
		active = false,
		disabled = false,
		onclick,
	}: {
		icon: Component<{ size?: number; strokeWidth?: number }>;
		label: string;
		shortcut?: string;
		active?: boolean;
		disabled?: boolean;
		onclick: () => void;
	} = $props();
</script>

<button
	class="grid size-8 place-items-center rounded-sm text-ink-dim transition-colors hover:bg-raised hover:text-ink disabled:opacity-30 disabled:hover:bg-transparent
		{active ? 'bg-hover text-ink shadow-[inset_0_0_0_1px_var(--color-line)]' : ''}"
	title={shortcut ? `${label} (${shortcut})` : label}
	aria-label={label}
	aria-pressed={active}
	{disabled}
	{onclick}
>
	<Icon size={17} strokeWidth={1.6} />
</button>
