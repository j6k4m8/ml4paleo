<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import type { ProjectImage } from "#lib/types.ts";
	import { ChunkStore } from "./chunks";
	import { absolute, loadLevels } from "./image";
	import { WorkerPool } from "./loader";
	import { PlaneRenderer } from "./plane";
	import { chooseLevel, type Level, tileId, type View, visibleTiles } from "./tiles";

	let { image }: { image: ProjectImage } = $props();

	const CACHE_BYTES = 512 * 1024 * 1024;

	let canvas: HTMLCanvasElement;
	let levels: Level[] = $state([]);
	let error = $state("");
	let z = $state(0);
	let centerX = $state(0);
	let centerY = $state(0);
	let zoom = $state(1);
	let width = $state(0);
	let height = $state(0);
	let low = $state(0);
	let high = $state(1);

	let renderer: PlaneRenderer | undefined;
	let pool: WorkerPool | undefined;
	let store: ChunkStore | undefined;
	let frame = 0;
	const controller = new AbortController();

	const depth = $derived(image.manifest.shape_czyx[1]);

	onMount(() => {
		[low, high] = image.manifest.window;
		const observer = new ResizeObserver(([entry]) => {
			if (!entry) return;
			const ratio = window.devicePixelRatio || 1;
			width = Math.round(entry.contentRect.width * ratio);
			height = Math.round(entry.contentRect.height * ratio);
			canvas.width = width;
			canvas.height = height;
		});
		observer.observe(canvas);
		(async () => {
			try {
				renderer = new PlaneRenderer(canvas);
				levels = await loadLevels(image.zarr_url, controller.signal);
				pool = new WorkerPool();
				store = new ChunkStore(pool.loader(absolute(image.zarr_url), levels), CACHE_BYTES);
				const [, nz, ny, nx] = image.manifest.shape_czyx;
				z = Math.floor(nz / 2);
				centerY = ny / 2;
				centerX = nx / 2;
				zoom = Math.min(width / nx, height / ny) || 1;
			} catch (e) {
				if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => observer.disconnect();
	});

	onDestroy(() => {
		controller.abort();
		cancelAnimationFrame(frame);
		store?.keepOnly(new Set());
		pool?.close();
		renderer?.destroy();
	});

	function view(): View {
		return { z, centerX, centerY, zoom, width, height };
	}

	function layers(current: View) {
		const coarsest = levels[levels.length - 1];
		const chosen = chooseLevel(levels, current.zoom);
		const order = coarsest && coarsest !== chosen ? [coarsest, chosen] : [chosen];
		return order.map((level) => ({ level, tiles: visibleTiles(level, current) }));
	}

	function schedule() {
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(render);
	}

	function render() {
		if (!renderer || !store || levels.length === 0 || width === 0) return;
		const current = view();
		const wanted = layers(current);
		store.keepOnly(new Set(wanted.flatMap(({ tiles }) => tiles.map(tileId))));
		for (const { level, tiles } of wanted) {
			for (const key of tiles) {
				if (renderer.has(key, current.z, level)) continue;
				const id = tileId(key);
				const cached = store.get(id);
				if (cached) {
					renderer.upload(key, current.z, level, cached);
					continue;
				}
				store.request(id).then(
					(chunk) => {
						if (z !== current.z) return;
						renderer?.upload(key, z, level, chunk);
						schedule();
					},
					(e: unknown) => {
						if (e instanceof DOMException && e.name === "AbortError") return;
						error = e instanceof Error ? e.message : String(e);
					},
				);
			}
		}
		renderer.draw(current, [low, high], wanted);
	}

	$effect(() => {
		// Redraw whenever the view, the window, or the levels change.
		void [z, centerX, centerY, zoom, width, height, low, high, levels.length];
		schedule();
	});

	let dragging: { x: number; y: number } | null = null;

	function pointerDown(event: PointerEvent) {
		dragging = { x: event.clientX, y: event.clientY };
		canvas.setPointerCapture(event.pointerId);
	}

	function pointerMove(event: PointerEvent) {
		if (!dragging) return;
		const ratio = window.devicePixelRatio || 1;
		centerX -= ((event.clientX - dragging.x) * ratio) / zoom;
		centerY -= ((event.clientY - dragging.y) * ratio) / zoom;
		dragging = { x: event.clientX, y: event.clientY };
	}

	function pointerUp() {
		dragging = null;
	}

	function wheel(event: WheelEvent) {
		event.preventDefault();
		const ratio = window.devicePixelRatio || 1;
		const rect = canvas.getBoundingClientRect();
		// Keep the voxel under the cursor in place.
		const px = (event.clientX - rect.left) * ratio - width / 2;
		const py = (event.clientY - rect.top) * ratio - height / 2;
		const next = Math.min(64, Math.max(1 / 256, zoom * Math.exp(-event.deltaY * 0.002)));
		centerX += px / zoom - px / next;
		centerY += py / zoom - py / next;
		zoom = next;
	}

	function key(event: KeyboardEvent) {
		const step = event.shiftKey ? 10 : 1;
		if (event.key === "ArrowUp" || event.key === ".") z = Math.min(depth - 1, z + step);
		else if (event.key === "ArrowDown" || event.key === ",") z = Math.max(0, z - step);
		else return;
		event.preventDefault();
	}
</script>

<div class="viewer">
	<canvas
		bind:this={canvas}
		tabindex="0"
		aria-label="Image plane {z + 1} of {depth}"
		onpointerdown={pointerDown}
		onpointermove={pointerMove}
		onpointerup={pointerUp}
		onpointercancel={pointerUp}
		onwheel={wheel}
		onkeydown={key}
	></canvas>
	<div class="controls">
		<label>
			Slice {z + 1} / {depth}
			<input type="range" min="0" max={depth - 1} bind:value={z} />
		</label>
		<label>
			Window
			<input type="number" step="any" bind:value={low} aria-label="Window low" />
			<input type="number" step="any" bind:value={high} aria-label="Window high" />
		</label>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
	</div>
</div>

<style>
	.viewer {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	canvas {
		flex: 1;
		width: 100%;
		min-height: 0;
		background: #000;
		touch-action: none;
		cursor: grab;
	}
	canvas:focus-visible {
		outline: 2px solid var(--accent);
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: center;
		padding: 0.5rem 0;
	}
	.controls input[type="number"] {
		width: 7rem;
	}
	.error {
		color: var(--danger);
	}
</style>
