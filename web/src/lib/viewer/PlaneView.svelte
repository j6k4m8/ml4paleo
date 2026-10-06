<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import type { ChunkStore } from "./chunks";
	import type { LabelLayer } from "./labels";
	import { type LabelTile, PlaneRenderer } from "./plane";
	import type { ViewerState } from "./state.svelte";
	import {
		chooseLevel,
		type Level,
		type Plane,
		pixelsPerVoxel,
		sliceIndex,
		type TileKey,
		tileId,
		type Vec3,
		type View,
		visibleTiles,
		voxelAt,
	} from "./tiles";

	let {
		plane,
		viewer,
		levels,
		images,
		labels,
		onhover,
		onresize,
	}: {
		plane: Plane;
		viewer: ViewerState;
		levels: Level[];
		images: ChunkStore;
		labels: LabelLayer | null;
		onhover: (plane: Plane) => void;
		onresize: (plane: Plane, width: number, height: number) => void;
	} = $props();

	// Labels are full resolution only until the label pyramid exists, so
	// zoomed far out they'd need too many chunks (each is 256 KiB).
	const MAX_LABEL_TILES = 256;
	const AXIS_NAMES = ["z", "y", "x"];

	let canvas: HTMLCanvasElement;
	let renderer: PlaneRenderer | undefined;
	let width = $state(0);
	let height = $state(0);
	let error = $state("");
	let labelsHidden = $state(false);
	let frame = 0;

	const slice = $derived(Math.floor(viewer.position[plane.normal]));

	// Loads this view is waiting for, so each gets one handler.
	const waiting = new Set<string>();

	function view(): View {
		return {
			plane,
			position: viewer.position,
			zoom: viewer.zoom,
			aspect: viewer.aspect,
			width,
			height,
			pixelRatio: window.devicePixelRatio || 1,
		};
	}

	function start() {
		renderer = new PlaneRenderer(canvas, plane, viewer.shape);
		if (labels) renderer.setPalette(labels.colors);
	}

	onMount(() => {
		const observer = new ResizeObserver(([entry]) => {
			if (!entry) return;
			const ratio = window.devicePixelRatio || 1;
			width = Math.max(1, Math.round(entry.contentRect.width * ratio));
			height = Math.max(1, Math.round(entry.contentRect.height * ratio));
			canvas.width = width;
			canvas.height = height;
			onresize(plane, width, height);
		});
		observer.observe(canvas);
		// The browser may drop the GPU context (driver reset, too many
		// contexts); start over with a fresh renderer when it comes back.
		const lost = (event: Event) => {
			event.preventDefault();
			renderer = undefined;
			waiting.clear();
		};
		const restored = () => {
			start();
			schedule();
		};
		canvas.addEventListener("webglcontextlost", lost);
		canvas.addEventListener("webglcontextrestored", restored);
		try {
			start();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		return () => {
			observer.disconnect();
			canvas.removeEventListener("webglcontextlost", lost);
			canvas.removeEventListener("webglcontextrestored", restored);
		};
	});

	onDestroy(() => {
		cancelAnimationFrame(frame);
		images.want(plane.name, new Set());
		labels?.store.want(plane.name, new Set());
		renderer?.destroy();
	});

	// New label colors, and chunks someone else just edited.
	$effect(() => {
		if (!labels) return;
		renderer?.setPalette(labels.colors);
		const stop = labels.onChange((ids) => {
			for (const id of ids) renderer?.dropLabels(id);
			schedule();
		});
		return stop;
	});

	$effect(() => {
		// Redraw whenever anything the view shows changes.
		void [viewer.position, viewer.zoom, viewer.window, viewer.opacity, viewer.showLabels, width, height, labels];
		schedule();
	});

	function schedule() {
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(render);
	}

	function failed(e: unknown) {
		if (e instanceof DOMException && e.name === "AbortError") return;
		error = e instanceof Error ? e.message : String(e);
	}

	function render() {
		if (!renderer || levels.length === 0 || width === 0) return;
		const current = view();
		const coarsest = levels[levels.length - 1]!;
		const chosen = chooseLevel(levels, current);
		const layers = (coarsest === chosen ? [chosen] : [coarsest, chosen]).map((level) => ({
			level,
			slice: sliceIndex(level, current),
			tiles: visibleTiles(level, current),
		}));
		const imageIds = new Set(layers.flatMap(({ tiles }) => tiles.map(tileId)));
		images.want(plane.name, imageIds);
		renderer.reserve(imageIds.size, MAX_LABEL_TILES);
		for (const { slice, tiles } of layers) {
			for (const key of tiles) loadImage(key, slice);
		}

		let labelLayer: { slice: number; tiles: LabelTile[]; opacity: number } | null = null;
		const full = levels[0]!;
		const fullTiles = viewer.showLabels && labels ? visibleTiles(full, current, 0) : [];
		labelsHidden = fullTiles.length > MAX_LABEL_TILES;
		if (labels && fullTiles.length > 0 && !labelsHidden) {
			const tiles = fullTiles.map((key) => ({ id: `${key.cz}/${key.cy}/${key.cx}`, key }));
			const labelSlice = sliceIndex(full, current);
			labels.store.want(plane.name, new Set(tiles.map((t) => t.id)));
			for (const tile of tiles) loadLabels(labels, tile, labelSlice);
			labelLayer = { slice: labelSlice, tiles, opacity: viewer.opacity };
		} else {
			labels?.store.want(plane.name, new Set());
		}
		renderer.draw(current, viewer.window, layers, labelLayer);
	}

	function loadImage(key: TileKey, at: number) {
		if (!renderer || renderer.hasImage(key, at)) return;
		const id = tileId(key);
		const cached = images.get(id);
		if (cached) return renderer.uploadImage(key, at, cached);
		if (waiting.has(`image:${id}`)) return;
		waiting.add(`image:${id}`);
		images
			.request(id)
			.then((chunk) => {
				if (!renderer || renderer.hasImage(key, at)) return;
				if (sliceIndex(levels[key.level]!, view()) !== at) return schedule();
				renderer.uploadImage(key, at, chunk);
				schedule();
			}, failed)
			.finally(() => waiting.delete(`image:${id}`));
	}

	function loadLabels(layer: LabelLayer, tile: LabelTile, at: number) {
		if (!renderer || renderer.hasLabels(tile.id, at)) return;
		const cached = layer.store.get(tile.id);
		if (cached) return renderer.uploadLabels(tile, at, cached);
		if (waiting.has(`labels:${tile.id}`)) return;
		waiting.add(`labels:${tile.id}`);
		layer.store
			.request(tile.id)
			.then((chunk) => {
				if (!renderer || renderer.hasLabels(tile.id, at)) return;
				if (sliceIndex(levels[0]!, view()) !== at) return schedule();
				renderer.uploadLabels(tile, at, chunk);
				schedule();
			}, failed)
			.finally(() => waiting.delete(`labels:${tile.id}`));
	}

	// --- pointer and wheel ---------------------------------------------------

	let press: { x: number; y: number; moved: boolean } | null = null;
	let wheelSteps = 0;

	function offset(event: MouseEvent): [number, number] {
		const ratio = window.devicePixelRatio || 1;
		const rect = canvas.getBoundingClientRect();
		return [(event.clientX - rect.left) * ratio - width / 2, (event.clientY - rect.top) * ratio - height / 2];
	}

	function pointerDown(event: PointerEvent) {
		press = { x: event.clientX, y: event.clientY, moved: false };
		canvas.setPointerCapture(event.pointerId);
		canvas.focus();
	}

	function pointerMove(event: PointerEvent) {
		onhover(plane);
		if (!press) return;
		const dx = event.clientX - press.x;
		const dy = event.clientY - press.y;
		if (!press.moved && Math.hypot(dx, dy) < 3) return;
		press.moved = true;
		viewer.autoFit = false;
		const ratio = window.devicePixelRatio || 1;
		const px = pixelsPerVoxel(view());
		const point = [...viewer.position] as Vec3;
		point[plane.u] -= (dx * ratio) / px[plane.u];
		point[plane.v] -= (dy * ratio) / px[plane.v];
		viewer.moveTo(point);
		press.x = event.clientX;
		press.y = event.clientY;
	}

	function pointerUp(event: PointerEvent) {
		if (press && !press.moved) {
			viewer.autoFit = false;
			viewer.moveTo(voxelAt(view(), ...offset(event)));
		}
		press = null;
	}

	function wheel(event: WheelEvent) {
		event.preventDefault();
		const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
		if (event.ctrlKey || event.metaKey) {
			viewer.autoFit = false;
			// Keep the voxel under the cursor in place.
			const [dx, dy] = offset(event);
			const under = voxelAt(view(), dx, dy);
			viewer.zoomBy(Math.exp(-delta * 0.002));
			const px = pixelsPerVoxel(view());
			const point = [...viewer.position] as Vec3;
			point[plane.u] = under[plane.u] - dx / px[plane.u];
			point[plane.v] = under[plane.v] - dy / px[plane.v];
			viewer.moveTo(point);
			return;
		}
		wheelSteps += delta / 40;
		const steps = Math.trunc(wheelSteps);
		if (steps !== 0) {
			wheelSteps -= steps;
			viewer.step(plane.normal, steps);
		}
	}
</script>

<div class="plane plane-{plane.normal}">
	<canvas
		bind:this={canvas}
		tabindex="0"
		aria-label="{plane.name.toUpperCase()} view at {AXIS_NAMES[plane.normal]} {slice}"
		onpointerdown={pointerDown}
		onpointermove={pointerMove}
		onpointerup={pointerUp}
		onpointercancel={() => (press = null)}
		onpointerenter={() => onhover(plane)}
		onwheel={wheel}
	></canvas>
	<div class="crosshair u axis-{plane.u}" aria-hidden="true"></div>
	<div class="crosshair v axis-{plane.v}" aria-hidden="true"></div>
	<div class="caption">
		{plane.name.toUpperCase()} · {AXIS_NAMES[plane.normal]}
		{slice}
		{#if labelsHidden}<span class="muted">· zoom in to see labels</span>{/if}
	</div>
	{#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<style>
	.plane {
		position: relative;
		min-width: 0;
		min-height: 0;
		border: 1px solid var(--axis-color);
		overflow: hidden;
	}
	.plane-0 {
		--axis-color: #539bf5;
	}
	.plane-1 {
		--axis-color: #57ab5a;
	}
	.plane-2 {
		--axis-color: #e5534b;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		background: #000;
		touch-action: none;
		cursor: crosshair;
	}
	canvas:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: -2px;
	}
	.crosshair {
		position: absolute;
		pointer-events: none;
		opacity: 0.6;
	}
	/* The line along u marks the plane of constant v, and vice versa. */
	.crosshair.u {
		left: 50%;
		top: 0;
		bottom: 0;
		width: 1px;
	}
	.crosshair.v {
		top: 50%;
		left: 0;
		right: 0;
		height: 1px;
	}
	.axis-0 {
		background: #539bf5;
	}
	.axis-1 {
		background: #57ab5a;
	}
	.axis-2 {
		background: #e5534b;
	}
	.caption {
		position: absolute;
		left: 0.4rem;
		top: 0.25rem;
		font-size: 0.8rem;
		color: #fff;
		text-shadow: 0 0 3px #000;
		pointer-events: none;
	}
	.caption .muted {
		color: #ccc;
	}
	.error {
		position: absolute;
		bottom: 0.25rem;
		left: 0.4rem;
		margin: 0;
		font-size: 0.8rem;
	}
</style>
