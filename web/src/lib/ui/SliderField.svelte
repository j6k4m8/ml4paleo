<script lang="ts">
	let {
		value = $bindable(),
		min,
		max,
		step = 1,
		label,
		numberLabel = label,
		class: extra = "",
	}: {
		value: number;
		min: number;
		max: number;
		step?: number;
		/** What it sets, shown beside it. */
		label: string;
		/** What the number box is called to screen readers. */
		numberLabel?: string;
		class?: string;
	} = $props();

	// The last number the box held, to show again if it is left empty.
	let last = value;
	$effect(() => {
		if (Number.isFinite(value)) last = value;
	});

	/** Leaving the box (or Enter) settles it: out of range is brought in, and an empty box shows the last number again. */
	function settle(event: Event & { currentTarget: HTMLInputElement }) {
		const typed = event.currentTarget.valueAsNumber;
		value = Number.isFinite(typed) ? Math.min(max, Math.max(min, typed)) : last;
		event.currentTarget.value = String(value);
	}
</script>

<!-- A slider and its number in one field, which a few of the options bar's settings are. -->
<label class="flex shrink-0 items-center gap-2 text-ink-dim {extra}">
	{label}
	<span class="flex h-6 items-center rounded-sm border border-edge bg-field focus-within:border-accent">
		<input class="mx-2 w-28" type="range" {min} {max} {step} bind:value />
		<input
			class="h-full w-20 rounded-none border-0 border-l border-edge bg-transparent px-1.5 font-mono text-xs text-ink focus:outline-none"
			type="number"
			{min}
			{max}
			{step}
			bind:value
			onchange={settle}
			aria-label={numberLabel}
		/>
	</span>
</label>
