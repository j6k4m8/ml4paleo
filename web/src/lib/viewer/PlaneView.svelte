<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import { PlaneMask } from "../labels/raster";
	import type { Roi } from "../rois.svelte";
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
		onstroke,
		onpolygon,
		rois,
		onroi,
	}: {
		plane: Plane;
		viewer: ViewerState;
		levels: Level[];
		images: ChunkStore;
		labels: LabelLayer | null;
		onhover: (plane: Plane) => void;
		onresize: (plane: Plane, width: number, height: number) => void;
		/** A finished brush or eraser stroke at level-0 `slice`. */
		onstroke: (plane: Plane, slice: number, mask: PlaneMask) => void;
		/** Close the polygon being drawn. */
		onpolygon: () => void;
		rois: Roi[];
		/** A rectangle drawn with the ROI tool, in plane voxels, at `slice`. */
		onroi: (plane: Plane, slice: number, corners: [[number, number], [number, number]]) => void;
	} = $props();

	// Labels are full resolution only until the label pyramid exists, so
	// zoomed far out they'd need too many chunks (each is 256 KiB; three
	// views share a 128 MiB cache).
	const MAX_LABEL_TILES = 128;
	// More image chunks than this in view, and the view uses a coarser level
	// (each chunk is 64³ voxels, so this bounds memory on big screens).
	const MAX_IMAGE_TILES = 400;
	const AXIS_NAMES = ["z", "y", "x"];

	let canvas: HTMLCanvasElement;
	let renderer: PlaneRenderer | undefined;
	let width = $state(0);
	let height = $state(0);
	let error = $state("");
	let labelsHidden = $state(false);
	let frame = 0;
	let destroyed = false;

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
		destroyed = true;
		cancelAnimationFrame(frame);
		images.want(plane.name, new Set());
		labels?.store.want(plane.name, new Set());
		renderer?.destroy();
		renderer = undefined;
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
		void [viewer.position, viewer.zoom, viewer.window[0], viewer.window[1], viewer.opacity, viewer.showLabels, width, height, labels];
		schedule();
	});

	function schedule() {
		if (destroyed) return;
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(render);
	}

	function failed(e: unknown) {
		if (e instanceof DOMException && e.name === "AbortError") return;
		error = e instanceof Error ? e.message : String(e);
	}

	function render() {
		if (destroyed || !renderer || levels.length === 0 || width === 0) return;
		const current = view();
		const coarsest = levels[levels.length - 1]!;
		let chosen = chooseLevel(levels, current);
		while (chosen !== coarsest && visibleTiles(chosen, current, 0).length > MAX_IMAGE_TILES) {
			chosen = levels[chosen.index + 1]!;
		}
		const layers = (coarsest === chosen ? [chosen] : [coarsest, chosen]).map((level) => ({
			level,
			slice: sliceIndex(level, current),
			tiles: visibleTiles(level, current),
		}));
		const imageIds = new Set(layers.flatMap(({ tiles }) => tiles.map(tileId)));
		// Tiles just outside the view load ahead but may be evicted.
		const shownIds = new Set(
			layers.flatMap(({ level }) => visibleTiles(level, current, 0).map(tileId)),
		);
		images.want(plane.name, imageIds, shownIds);
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

	let press: { x: number; y: number; moved: boolean; pan: boolean } | null = null;
	let stroke: { mask: PlaneMask; last: [number, number] } | null = null;
	let rectangle: { from: [number, number]; to: [number, number] } | null = $state(null);
	let cursor: [number, number] | null = $state(null);
	let overlay: HTMLCanvasElement;
	let wheelSteps = 0;

	const ratio = () => window.devicePixelRatio || 1;
	const painting = $derived(viewer.tool === "brush" || viewer.tool === "eraser");

	function offset(event: MouseEvent): [number, number] {
		const rect = canvas.getBoundingClientRect();
		return [(event.clientX - rect.left) * ratio() - width / 2, (event.clientY - rect.top) * ratio() - height / 2];
	}

	/** The plane point (u, v) under the pointer, in level-0 voxels. */
	function planePoint(event: MouseEvent): [number, number] {
		const point = voxelAt(view(), ...offset(event));
		return [point[plane.u], point[plane.v]];
	}

	/** Where a plane point is on screen, in CSS pixels. */
	function screen(u: number, v: number): [number, number] {
		const px = pixelsPerVoxel(view());
		return [
			((u - viewer.position[plane.u]) * px[plane.u] + width / 2) / ratio(),
			((v - viewer.position[plane.v]) * px[plane.v] + height / 2) / ratio(),
		];
	}

	/** Brush radii along u and v, in level-0 voxels. */
	function radii(): [number, number] {
		return [viewer.brushRadius / viewer.aspect[plane.u], viewer.brushRadius / viewer.aspect[plane.v]];
	}

	function canEdit(): boolean {
		return viewer.tool === "eraser" || viewer.tool === "roi" || viewer.activeClass !== null;
	}

	function pointerDown(event: PointerEvent) {
		canvas.setPointerCapture(event.pointerId);
		canvas.focus();
		const pan = viewer.tool === "navigate" || viewer.panning || event.button === 1;
		press = { x: event.clientX, y: event.clientY, moved: false, pan };
		if (pan || event.button !== 0 || !canEdit()) return;
		if (painting) {
			const point = planePoint(event);
			const mask = new PlaneMask(viewer.shape[plane.u], viewer.shape[plane.v]);
			mask.stamp(...point, ...radii());
			stroke = { mask, last: point };
			drawStroke();
		} else if (viewer.tool === "roi") {
			const point = planePoint(event);
			rectangle = { from: point, to: point };
		} else if (viewer.tool === "polygon") {
			const point = planePoint(event);
			const current = viewer.polygon;
			if (current && current.plane === plane.name && current.slice === slice) {
				viewer.polygon = { ...current, points: [...current.points, point] };
			} else {
				viewer.polygon = { plane: plane.name, slice, points: [point] };
			}
		}
	}

	function pointerMove(event: PointerEvent) {
		onhover(plane);
		cursor = offset(event).map((d, i) => (d + (i === 0 ? width : height) / 2) / ratio()) as [number, number];
		if (rectangle) {
			rectangle = { ...rectangle, to: planePoint(event) };
			return;
		}
		if (stroke) {
			const point = planePoint(event);
			stroke.mask.line(stroke.last, point, ...radii());
			stroke.last = point;
			drawStroke();
			return;
		}
		if (!press?.pan) return;
		const dx = event.clientX - press.x;
		const dy = event.clientY - press.y;
		if (!press.moved && Math.hypot(dx, dy) < 3) return;
		press.moved = true;
		viewer.autoFit = false;
		const px = pixelsPerVoxel(view());
		const point = [...viewer.position] as Vec3;
		point[plane.u] -= (dx * ratio()) / px[plane.u];
		point[plane.v] -= (dy * ratio()) / px[plane.v];
		viewer.moveTo(point);
		press.x = event.clientX;
		press.y = event.clientY;
	}

	function pointerUp(event: PointerEvent) {
		if (rectangle) {
			const { from, to } = rectangle;
			rectangle = null;
			onroi(plane, slice, [from, to]);
		} else if (stroke) {
			const finished = stroke.mask;
			stroke = null;
			clearStroke();
			if (finished.count > 0) onstroke(plane, slice, finished);
		} else if (press?.pan && !press.moved && viewer.tool === "navigate") {
			viewer.autoFit = false;
			viewer.moveTo(voxelAt(view(), ...offset(event)));
		}
		press = null;
	}

	function cancel() {
		press = null;
		stroke = null;
		rectangle = null;
		clearStroke();
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
		if (stroke) return;
		// About one slice per mouse wheel notch; trackpads add up.
		wheelSteps += delta / 100;
		const steps = Math.trunc(wheelSteps);
		if (steps !== 0) {
			wheelSteps -= steps;
			viewer.step(plane.normal, steps);
		}
	}

	// --- previews --------------------------------------------------------------

	const activeColor = $derived(
		viewer.tool === "eraser"
			? "#ffffff"
			: (labels?.classes.find((c) => c.value === viewer.activeClass)?.color ?? "#ffffff"),
	);

	function clearStroke() {
		overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
	}

	/** Paint the stroke so far: one rectangle per run of voxels in a row. */
	function drawStroke() {
		if (!stroke || !overlay) return;
		overlay.width = width;
		overlay.height = height;
		const context = overlay.getContext("2d");
		if (!context) return;
		const { mask } = stroke;
		const px = pixelsPerVoxel(view());
		const r = ratio();
		context.globalAlpha = Math.max(0.35, viewer.opacity);
		context.fillStyle = activeColor;
		for (let j = mask.v0; j < mask.v0 + mask.height; j++) {
			let i = mask.u0;
			while (i < mask.u0 + mask.width) {
				if (!mask.has(i, j)) {
					i++;
					continue;
				}
				let end = i;
				while (end < mask.u0 + mask.width && mask.has(end, j)) end++;
				const [x, y] = screen(i, j);
				context.fillRect(x * r, y * r, (end - i) * px[plane.u], px[plane.v]);
				i = end;
			}
		}
	}

	const polygonHere = $derived(
		viewer.polygon && viewer.polygon.plane === plane.name && viewer.polygon.slice === slice ? viewer.polygon : null,
	);

	function polygonPath(points: [number, number][], extra: [number, number] | null): string {
		void [viewer.position, viewer.zoom, width, height];
		const all = extra ? [...points.map(([u, v]) => screen(u, v)), extra] : points.map(([u, v]) => screen(u, v));
		return all.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
	}

	/** Screen rectangles of the ROIs this view's plane cuts through. */
	const roiOutlines = $derived.by(() => {
		void [viewer.position, viewer.zoom, width, height];
		const { normal, u, v } = plane;
		return rois
			.filter((roi) => roi.bbox[normal]! <= slice && slice < roi.bbox[normal + 3]!)
			.map((roi) => {
				const [x0, y0] = screen(roi.bbox[u]!, roi.bbox[v]!);
				const [x1, y1] = screen(roi.bbox[u + 3]!, roi.bbox[v + 3]!);
				return { id: roi.id, status: roi.status, x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
			});
	});

	const rectangleOutline = $derived.by(() => {
		void [viewer.position, viewer.zoom, width, height];
		if (!rectangle) return null;
		const [x0, y0] = screen(...rectangle.from);
		const [x1, y1] = screen(...rectangle.to);
		return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
	});

	const brushOutline = $derived.by(() => {
		void [viewer.position, viewer.zoom, viewer.brushRadius, width, height];
		if (!painting || !cursor) return null;
		const px = pixelsPerVoxel(view());
		const [ru, rv] = radii();
		return { x: cursor[0], y: cursor[1], rx: (ru * px[plane.u]) / ratio(), ry: (rv * px[plane.v]) / ratio() };
	});
</script>

<div class="plane plane-{plane.normal}">
	<canvas
		bind:this={canvas}
		tabindex="0"
		aria-label="{plane.name.toUpperCase()} view at {AXIS_NAMES[plane.normal]} {slice}"
		onpointerdown={pointerDown}
		onpointermove={pointerMove}
		onpointerup={pointerUp}
		onpointercancel={cancel}
		onpointerenter={() => onhover(plane)}
		onpointerleave={() => (cursor = null)}
		ondblclick={() => viewer.tool === "polygon" && onpolygon()}
		onfocus={() => onhover(plane)}
		onwheel={wheel}
		class:editing={viewer.tool !== "navigate" && !viewer.panning}
	></canvas>
	<canvas class="overlay" bind:this={overlay} aria-hidden="true"></canvas>
	<svg class="overlay" aria-hidden="true">
		{#each roiOutlines as roi (roi.id)}
			<rect
				class="roi roi-{roi.status}"
				class:selected={roi.id === viewer.selectedRoi}
				x={roi.x}
				y={roi.y}
				width={Math.max(1, roi.w)}
				height={Math.max(1, roi.h)}
			/>
		{/each}
		{#if rectangleOutline}
			<rect class="roi roi-new" x={rectangleOutline.x} y={rectangleOutline.y} width={rectangleOutline.w} height={rectangleOutline.h} />
		{/if}
		{#if polygonHere}
			<polyline
				points={polygonPath(polygonHere.points, cursor)}
				fill={activeColor}
				fill-opacity="0.25"
				stroke={activeColor}
				stroke-width="1.5"
			/>
		{/if}
		{#if brushOutline}
			<ellipse
				cx={brushOutline.x}
				cy={brushOutline.y}
				rx={Math.max(1, brushOutline.rx)}
				ry={Math.max(1, brushOutline.ry)}
				fill="none"
				stroke={activeColor}
				stroke-width="1"
			/>
		{/if}
	</svg>
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
		cursor: grab;
	}
	canvas.editing {
		cursor: crosshair;
	}
	.roi {
		fill: none;
		stroke-width: 1.5;
		stroke-dasharray: 6 3;
	}
	.roi-open {
		stroke: #e3b341;
	}
	.roi-complete {
		stroke: #57ab5a;
		stroke-dasharray: none;
	}
	.roi-skipped {
		stroke: #768390;
	}
	.roi-new {
		stroke: #fff;
	}
	.roi.selected {
		stroke-width: 3;
	}
	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		background: none;
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
