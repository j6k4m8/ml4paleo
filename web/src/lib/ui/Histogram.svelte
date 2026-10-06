<script lang="ts">
	let {
		counts,
		edges,
		window = $bindable(),
	}: {
		counts: number[];
		edges: number[];
		window: [number, number];
	} = $props();

	// Levels-style: the image's histogram (log scale), with the display
	// window's ends as handles to drag. It's for the pointer only; the Black
	// and White fields next to it set the same values.
	const WIDTH = 220;
	const HEIGHT = 56;
	let svg: SVGSVGElement;
	let dragging: 0 | 1 | null = null;

	const lo = $derived(edges[0] ?? 0);
	const hi = $derived(edges[edges.length - 1] ?? 1);
	const path = $derived.by(() => {
		const top = Math.max(1, ...counts.map((c) => Math.log1p(c)));
		const step = WIDTH / Math.max(1, counts.length);
		let d = `M0,${HEIGHT}`;
		counts.forEach((c, i) => {
			const y = HEIGHT - (Math.log1p(c) / top) * (HEIGHT - 2);
			d += ` L${i * step},${y} L${(i + 1) * step},${y}`;
		});
		return `${d} L${WIDTH},${HEIGHT} Z`;
	});

	const x = (value: number) => ((value - lo) / (hi - lo || 1)) * WIDTH;
	const valueAt = (clientX: number) => {
		const rect = svg.getBoundingClientRect();
		const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
		return lo + fraction * (hi - lo);
	};

	function down(event: PointerEvent) {
		const value = valueAt(event.clientX);
		dragging = Math.abs(value - window[0]) <= Math.abs(value - window[1]) ? 0 : 1;
		svg.setPointerCapture(event.pointerId);
		move(event);
	}

	function move(event: PointerEvent) {
		if (dragging === null) return;
		const value = Math.round(valueAt(event.clientX) * 1000) / 1000;
		window = dragging === 0 ? [Math.min(value, window[1]), window[1]] : [window[0], Math.max(value, window[0])];
	}
</script>

<svg
	bind:this={svg}
	viewBox="0 0 {WIDTH} {HEIGHT + 8}"
	class="w-full cursor-ew-resize touch-none select-none"
	aria-hidden="true"
	onpointerdown={down}
	onpointermove={move}
	onpointerup={() => (dragging = null)}
>
	<rect width={WIDTH} height={HEIGHT} fill="var(--color-field)" />
	<rect x={x(window[0])} width={Math.max(0, x(window[1]) - x(window[0]))} height={HEIGHT} fill="var(--color-accent-soft)" opacity="0.6" />
	<path d={path} fill="var(--color-ink-dim)" opacity="0.8" />
	{#each [window[0], window[1]] as value, i (i)}
		<line x1={x(value)} x2={x(value)} y1="0" y2={HEIGHT} stroke="var(--color-accent-hover)" stroke-width="1" />
		<path d="M{x(value)},{HEIGHT} l-4,7 h8 z" fill={i === 0 ? "var(--color-edge)" : "var(--color-ink)"} stroke="var(--color-accent-hover)" stroke-width="0.8" />
	{/each}
</svg>
