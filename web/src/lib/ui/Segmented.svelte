<script lang="ts" generics="T extends string">
	import type { Component } from "svelte";
	import { tooltip } from "./tooltip";

	interface Option {
		value: T;
		label: string;
		icon?: Component<{ size?: number }>;
		/** What it does, as a tooltip. */
		title?: string;
	}

	let {
		value,
		options,
		label,
		title,
		onchange,
		class: extra = "",
	}: {
		value: T;
		options: readonly Option[];
		/** What the group is, for screen readers. */
		label: string;
		/** A tooltip for the group, where its buttons have none to give. */
		title?: string;
		onchange: (value: T) => void;
		class?: string;
	} = $props();
</script>

<!-- A row of buttons, one of them pressed: for choosing among a few things in an options bar. -->
<div class="inline-flex h-6 shrink-0 divide-x divide-edge rounded-sm border border-edge {extra}" role="group" aria-label={label} use:tooltip={title}>
	{#each options as option (option.value)}
		<button
			type="button"
			class="relative flex items-center gap-1 px-2 whitespace-nowrap first:rounded-l-[3px] last:rounded-r-[3px] focus-visible:z-10
				{value === option.value ? 'bg-accent-fill text-white' : 'bg-raised text-ink-dim hover:bg-hover hover:text-ink'}"
			aria-pressed={value === option.value}
			use:tooltip={option.title}
			onclick={() => onchange(option.value)}
		>
			{#if option.icon}<option.icon size={12} />{/if}
			{option.label}
		</button>
	{/each}
</div>
