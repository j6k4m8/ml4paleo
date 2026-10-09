<script lang="ts">
	import ChevronDown from "@lucide/svelte/icons/chevron-down";
	import { summarize } from "#lib/labels/modes.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";

	let {
		classes,
		selected,
		label,
		onchange,
	}: {
		/** What there is to choose from, background first. */
		classes: readonly LabelClass[];
		/** The values chosen; at least one stays chosen. */
		selected: readonly number[];
		/** What the choice is, for the menu's heading and screen readers. */
		label: string;
		onchange: (values: number[]) => void;
	} = $props();

	const id = $props.id();
	let button: HTMLButtonElement | undefined = $state();
	let menu: HTMLElement | undefined = $state();
	let open = $state(false);

	const chosen = $derived(classes.filter((c) => selected.includes(c.value)));
	const names = $derived(chosen.map((c) => c.name));

	/** Check or uncheck a class; the last one checked can't be unchecked. */
	function pick(value: number, checkbox: HTMLInputElement) {
		const next = checkbox.checked ? [...selected, value] : selected.filter((v) => v !== value);
		if (next.length === 0) {
			checkbox.checked = true;
			return;
		}
		onchange(classes.map((c) => c.value).filter((v) => next.includes(v)));
	}

	// The menu is in the top layer, so it isn't cut off by the options bar
	// that scrolls; it goes under its button, kept inside the window.
	function placeBelowButton(event: ToggleEvent) {
		if (event.newState !== "open" || !button || !menu) return;
		const rect = button.getBoundingClientRect();
		menu.style.top = `${rect.bottom + 4}px`;
		menu.style.left = `${rect.left}px`;
	}

	function opened(event: ToggleEvent) {
		open = event.newState === "open";
		if (!open || !button || !menu) return;
		const room = window.innerWidth - 8 - menu.offsetWidth;
		menu.style.left = `${Math.max(8, Math.min(button.getBoundingClientRect().left, room))}px`;
		// Into the menu, so keys work there and Esc closes just it.
		menu.querySelector<HTMLInputElement>("input:checked")?.focus();
	}

	/** Esc closes the menu, and isn't the viewer's Esc, which would leave the tool. */
	function keepsEscape(node: HTMLElement) {
		const onKey = (event: KeyboardEvent) => {
			if (event.key === "Escape") event.stopPropagation();
		};
		node.addEventListener("keydown", onKey);
		return () => node.removeEventListener("keydown", onKey);
	}
</script>

<svelte:window onresize={() => menu?.hidePopover()} />

<button
	bind:this={button}
	type="button"
	class="btn max-w-56 shrink-0 justify-start gap-1.5 pr-1.5"
	popovertarget={id}
	aria-haspopup="true"
	aria-expanded={open}
	aria-label="{label}: {names.join(', ')}"
	title={`${label}: ${names.join(", ")}`}
>
	<span class="flex shrink-0 -space-x-1">
		{#each chosen.slice(0, 3) as c (c.value)}
			<span class="size-3 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={c.color}></span>
		{/each}
	</span>
	<span class="truncate">{summarize(names)}</span>
	<ChevronDown size={12} class="shrink-0 text-ink-dim" />
</button>

<div
	{id}
	bind:this={menu}
	popover="auto"
	role="group"
	aria-label={label}
	class="m-0 max-h-80 min-w-48 max-w-72 overflow-y-auto rounded-sm border border-line bg-panel p-1 text-xs text-ink shadow-xl shadow-black/50"
	onbeforetoggle={placeBelowButton}
	ontoggle={opened}
	{@attach keepsEscape}
>
	<p class="px-2 pt-1 pb-1.5 text-2xs font-semibold tracking-wide text-ink-dim uppercase">{label}</p>
	{#each classes as c (c.value)}
		<label class="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1 hover:bg-raised">
			<input type="checkbox" checked={selected.includes(c.value)} onchange={(event) => pick(c.value, event.currentTarget)} />
			<span class="size-3 shrink-0 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={c.color}></span>
			<span class="min-w-0 flex-1 truncate">{c.name}</span>
		</label>
	{/each}
	<p class="px-2 pt-1 pb-0.5 text-2xs text-ink-faint">At least one stays checked.</p>
</div>
