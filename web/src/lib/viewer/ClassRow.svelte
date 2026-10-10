<script lang="ts">
	import ChevronRight from "@lucide/svelte/icons/chevron-right";
	import Eye from "@lucide/svelte/icons/eye";
	import EyeOff from "@lucide/svelte/icons/eye-off";
	import { message } from "#lib/api.ts";
	import type { LabelClass } from "./labels";
	import type { ClassDisplay } from "./class-display.svelte";

	let { label, index, active, display, disabled, onselect, oncolor, onsave }: {
		label: LabelClass;
		index: number;
		active: boolean;
		display: ClassDisplay;
		disabled: boolean;
		onselect: () => void;
		oncolor: (color: string) => void;
		onsave: () => Promise<void>;
	} = $props();
	let expanded = $state(false);
	let error = $state("");
	let saving = $state(0);
	const visible = $derived(display.styles[label.value]?.visible !== false);
	const opacity = $derived(Math.round((display.styles[label.value]?.opacity ?? 1) * 100));
	const optionsId = $props.id();

	async function save() {
		error = "";
		saving++;
		try { await onsave(); } catch (e) { error = message(e); }
		finally { saving--; }
	}
</script>

<li>
	<div class="flex items-center gap-1 px-1 py-0.5 {active ? 'bg-accent-soft text-ink' : ''}">
		<button class="btn btn-ghost !px-1" aria-label="{label.name} display options" aria-expanded={expanded}
			aria-controls={optionsId} onclick={() => expanded = !expanded}>
			<ChevronRight size={13} class={expanded ? "rotate-90" : ""} />
		</button>
		<button class="flex min-w-0 flex-1 items-center gap-2 py-1 text-left" aria-pressed={active} onclick={onselect}>
			<span class="size-3 shrink-0 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={label.color}></span>
			<span class="min-w-0 flex-1 truncate {visible && opacity > 0 ? '' : 'text-ink-dim'}">{label.name}</span>
			{#if !visible}<span class="text-2xs text-ink-dim">hidden</span>
			{:else if opacity < 100}<span class="text-2xs text-ink-dim">{opacity}%</span>{/if}
			{#if index < 9}<span class="kbd">{index + 1}</span>{/if}
		</button>
		<button class="btn btn-ghost !px-1.5" {disabled} aria-label="{visible ? 'Hide' : 'Show'} {label.name}"
			title="{visible ? 'Hide' : 'Show'} {label.name} in labels, predictions, and 3D"
			onclick={() => display.set(label.value, { visible: !visible })}>
			{#if visible}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
		</button>
	</div>
	{#if expanded}
		<div id={optionsId} class="mx-2.5 mb-2 flex flex-col gap-2 border-l border-edge py-2 pl-3">
			<label class="flex items-center gap-2">
				<span class="w-12 shrink-0 text-ink-dim">Opacity</span>
				<input type="range" min="0" max="100" step="1" value={opacity} {disabled}
					class="min-w-0 flex-1" aria-label="{label.name} opacity"
					oninput={(event) => display.set(label.value, { opacity: Number(event.currentTarget.value) / 100 })} />
				<output class="w-8 text-right tabular-nums">{opacity}%</output>
			</label>
			<label class="flex items-center gap-2">
				<span class="w-12 shrink-0 text-ink-dim">Color</span>
				<input type="color" class="h-6 w-8 cursor-pointer rounded-sm border border-edge bg-field p-0.5"
					value={label.color} {disabled} aria-label="{label.name} color"
					oninput={(event) => oncolor(event.currentTarget.value)} onchange={save} />
				<span class="text-2xs text-ink-dim">{label.color}</span>
				{#if saving}<span class="text-2xs text-ink-dim" role="status">Saving…</span>{/if}
			</label>
			{#if !visible}<p class="text-2xs text-ink-dim">Hidden; show this class to see opacity or color changes.</p>{/if}
			<button class="btn btn-ghost self-start !px-0" {disabled} onclick={() => display.set(label.value, { visible: true, opacity: 1 })}>
				Reset visibility &amp; opacity
			</button>
		</div>
	{/if}
	{#if error}<p class="error px-2.5 pb-2" role="alert">{error}</p>{/if}
</li>
