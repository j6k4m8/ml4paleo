<script lang="ts">
	import { onDestroy, onMount, untrack } from "svelte";
	import LoaderCircle from "@lucide/svelte/icons/loader-circle";
	import Rotate3d from "@lucide/svelte/icons/rotate-3d";
	import { tooltip } from "#lib/ui/tooltip.ts";
	import type { LabelLayer } from "./labels";
	import type { ClassStyles } from "./class-display";
	import type { ChunkStore } from "./chunks";
	import { minimapIds, minimapLevel, minimapSuggestionIds, MinimapRenderer, type MinimapChunk, type MinimapSuggestionArea } from "./minimap";
	import { MinimapMesher } from "./minimap-meshing";
	import type { ViewerState } from "./state.svelte";

	let { project, viewer, labels, classStyles = {}, suggestions = [], onpanstart }: {
		project: string; viewer: ViewerState; labels: LabelLayer | null; classStyles?: ClassStyles;
		suggestions?: readonly (MinimapSuggestionArea & { store: ChunkStore })[]; onpanstart?: () => void;
	} = $props();

	let canvas: HTMLCanvasElement;
	let renderer: MinimapRenderer | null = null;
	let mesher: MinimapMesher | null = null;
	let meshErrors = $state({ labels: "", suggestions: "" });
	let meshDetail = $state({ labels: 1, suggestions: 1 });
	const detail = $derived(Math.max(meshDetail.labels, viewer.showPrediction ? meshDetail.suggestions : 1));
	let error = $state("");
	let loaded = $state(0);
	let total = $state(0);
	let levelName = $state("");
	let revision = $state(0);
	let suggested = $state(0);
	let frame = 0;
	let dragging: { id: number; x: number; y: number; panning: boolean } | null = $state(null);
	const owner = "minimap";

	function retryChunks() {
		labels?.store.retryFailed();
		for (const source of suggestions) source.store.retryFailed();
		error = "";
		revision++;
	}

	function schedule() {
		if (frame || !renderer) return;
		frame = requestAnimationFrame(() => {
			frame = 0;
			renderer?.draw(viewer.position, {
				labels: viewer.showLabels ? viewer.opacity : 0,
				suggestions: viewer.showPrediction ? viewer.opacity : 0,
			});
		});
	}

	function resize() {
		const ratio = window.devicePixelRatio || 1;
		const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
		const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
		if (canvas.width !== width) canvas.width = width;
		if (canvas.height !== height) canvas.height = height;
		schedule();
	}

	onMount(() => {
		const observer = new ResizeObserver(resize);
		observer.observe(canvas);
		try {
			renderer = new MinimapRenderer(canvas, viewer.shape, viewer.aspect);
			mesher = new MinimapMesher(project, (reply) => {
				meshErrors[reply.layer] = reply.error ?? "";
				// A failed/in-flight replacement must not erase usable geometry.
				if (reply.data) renderer?.setMesh(reply.layer, reply.data);
				if (reply.downsample) meshDetail[reply.layer] = reply.downsample;
				suggested = renderer?.suggestionCount ?? 0;
				schedule();
			});
			resize();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		const lost = (event: Event) => {
			event.preventDefault();
			renderer = null;
		};
		const restored = () => {
			try {
				renderer = new MinimapRenderer(canvas, viewer.shape, viewer.aspect);
				revision++;
			} catch (e) {
				error = e instanceof Error ? e.message : String(e);
			}
		};
		canvas.addEventListener("webglcontextlost", lost);
		canvas.addEventListener("webglcontextrestored", restored);
		return () => {
			observer.disconnect();
			canvas.removeEventListener("webglcontextlost", lost);
			canvas.removeEventListener("webglcontextrestored", restored);
		};
	});

	onDestroy(() => {
		cancelAnimationFrame(frame);
		labels?.store.want(owner, []);
		mesher?.destroy();
		renderer?.destroy();
		renderer = null;
	});

	// Position and camera changes redraw without rebuilding surfaces.
	$effect(() => {
		viewer.position;
		viewer.showLabels;
		viewer.showPrediction;
		viewer.opacity;
		schedule();
	});

	// Keep the selected coarse level resident. LabelLayer refreshes wanted
	// coarse chunks after edits; rebuilding from its cache then makes the 3D
	// view follow the pen after the server has updated the pyramid.
	$effect(() => {
		revision;
		const styles = classStyles;
		const layer = labels;
		const level = minimapLevel(layer?.levels ?? []);
		meshErrors.labels = "";
		schedule();
		if (!layer || !level) { renderer?.setMesh("labels", new Float32Array()); return; }
		let stopped = false;
		let pending = 0;
		const ids = minimapIds(level);
		const wanted = new Set(ids);
		total = ids.length;
		loaded = 0;
		levelName = level.index === 0 ? "full resolution" : `level ${level.index}`;
		layer.store.want(owner, ids);

		const rebuild = () => {
			if (!pending && !stopped) pending = requestAnimationFrame(build);
		};
		const build = () => {
			pending = 0;
			if (stopped || !renderer) return;
			const chunks: MinimapChunk[] = ids.flatMap((id) => {
				const chunk = layer.store.peek(id);
				return chunk ? [{ id, chunk }] : [];
			});
			loaded = chunks.length;
			if (chunks.length !== ids.length) return;
			mesher?.request("labels", { level, chunks, colors: layer.colors, shape: viewer.shape, aspect: viewer.aspect, styles: $state.snapshot(styles) });
		};
		for (const id of ids) layer.store.request(id).then(rebuild, (e: unknown) => {
			if (!stopped && !(e instanceof DOMException && e.name === "AbortError")) error = "Some overview labels couldn't load.";
		});
		const offChange = layer.onChange((changed) => {
			if (changed.some((id) => wanted.has(id))) rebuild();
		});
		const offLevels = layer.onLevels(() => revision++);
		const offClasses = layer.onClasses(rebuild);
		rebuild();
		return () => {
			stopped = true;
			mesher?.cancel("labels");
			cancelAnimationFrame(pending);
			offChange();
			offLevels();
			offClasses();
			layer.store.want(owner, []);
		};
	});

	// Only request ready artifacts near the shared center, never new predictions.
	// A primitive key prevents sub-voxel pan motion from rebuilding the same data.
	const suggestionIds = $derived(viewer.showPrediction && viewer.opacity > 0
		? minimapSuggestionIds(viewer.shape, viewer.position, suggestions).join("|") : "");
	$effect(() => {
		revision;
		const layer = labels, sources = suggestions, styles = classStyles;
		const ids = suggestionIds ? suggestionIds.split("|") : [];
		const maskOwner = "minimap:suggestions";
		let stopped = false, pending = 0;
		suggested = 0;
		meshErrors.suggestions = "";
		schedule();
		if (!layer || !ids.length) { renderer?.setMesh("suggestions", new Float32Array()); return; }
		const wanted = new Set(ids);
		const rebuild = () => { if (!pending && !stopped) pending = requestAnimationFrame(build); };
		const build = () => {
			pending = 0;
			if (stopped || !renderer) return;
			if (ids.some((id) => !layer.store.peek(id))) return;
			for (const source of sources) {
				const allowed = new Set(untrack(() => minimapSuggestionIds(viewer.shape, viewer.position, [source])));
				if (ids.some((id) => allowed.has(id) && !source.store.peek(id))) return;
			}
			const cached = (store: ChunkStore) => new Map(ids.flatMap((id) => {
				const chunk = store.peek(id);
				return chunk ? [[id, chunk] as const] : [];
			}));
			mesher?.request("suggestions", {
				level: { index: 0, path: "class", shape: viewer.shape, scale: [1, 1, 1] },
				chunks: [...cached(layer.store)].map(([id, chunk]) => ({ id, chunk })),
				colors: layer.colors, shape: viewer.shape, aspect: viewer.aspect, styles: $state.snapshot(styles),
			}, sources.map((source) => ({ box: [...source.box], chunks: cached(source.store) })), ids);
		};
		const load = (store: ChunkStore, keys: string[]) => {
			store.want(maskOwner, keys);
			for (const id of keys) store.request(id).then(rebuild, (e: unknown) => {
				if (!stopped && !(e instanceof DOMException && e.name === "AbortError")) error = "Some 3D suggestions couldn't load.";
			});
		};
		load(layer.store, ids);
		for (const source of sources) {
			const allowed = new Set(untrack(() => minimapSuggestionIds(viewer.shape, viewer.position, [source])));
			load(source.store, ids.filter((id) => allowed.has(id)));
		}
		const offChange = layer.onChange((changed) => { if (changed.some((id) => wanted.has(id))) rebuild(); });
		const offClasses = layer.onClasses(rebuild);
		rebuild();
		return () => {
			stopped = true;
			mesher?.cancel("suggestions");
			cancelAnimationFrame(pending);
			offChange();
			offClasses();
			layer.store.want(maskOwner, []);
			for (const source of sources) source.store.want(maskOwner, []);
		};
	});

	function down(event: PointerEvent) {
		if (event.button !== 0 || viewer.lassoing) return;
		event.preventDefault();
		dragging = { id: event.pointerId, x: event.clientX, y: event.clientY, panning: false };
		canvas.setPointerCapture(event.pointerId);
	}

	function move(event: PointerEvent) {
		if (!dragging || dragging.id !== event.pointerId || !renderer) return;
		const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y;
		if (event.shiftKey) {
			if (!dragging.panning) onpanstart?.();
			viewer.autoFit = false;
			viewer.moveTo(renderer.pan(viewer.position, dx, dy));
		} else renderer.orbit(dx, dy);
		dragging = { id: dragging.id, x: event.clientX, y: event.clientY, panning: event.shiftKey };
		schedule();
	}

	function up(event: PointerEvent) {
		if (dragging?.id === event.pointerId) dragging = null;
	}

	function wheel(event: WheelEvent) {
		event.preventDefault();
		renderer?.zoom(event.deltaY);
		schedule();
	}

	function navigate(event: MouseEvent) {
		event.preventDefault();
		if (!renderer || viewer.lassoing) return;
		const bounds = canvas.getBoundingClientRect();
		const point = renderer.pick(viewer.position, event.clientX - bounds.left, event.clientY - bounds.top, {
			labels: viewer.showLabels ? viewer.opacity : 0,
			suggestions: viewer.showPrediction ? viewer.opacity : 0,
		});
		if (!point) return;
		onpanstart?.();
		viewer.autoFit = false;
		viewer.moveTo(point);
		schedule();
	}
</script>

<div class="relative min-h-0 overflow-hidden bg-pasteboard" aria-label="3D minimap">
	<canvas
		bind:this={canvas}
		class="size-full touch-none"
		class:cursor-grabbing={!!dragging}
		class:cursor-grab={!dragging}
		aria-label="3D overview of saved labels, nearby suggestions, and current slices"
		use:tooltip={"Drag to orbit · Shift-drag to pan · wheel to zoom · right-click a surface to go there"}
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={up}
		onlostpointercapture={up}
		onwheel={wheel}
		oncontextmenu={navigate}
	></canvas>
	<div class="pointer-events-none absolute inset-x-2 top-2 flex items-center justify-between gap-2 text-2xs text-ink-dim">
		<span class="flex items-center gap-1"><Rotate3d size={13} /> 3D minimap</span>
		{#if total > 0 && loaded < total}<span class="flex items-center gap-1"><LoaderCircle size={12} class="animate-spin" /> {loaded}/{total}</span>{/if}
	</div>
	{#if suggested > 0 && viewer.showPrediction}
		<div class="pointer-events-none absolute top-7 left-2 text-2xs text-ink-dim">Striped: nearby suggestions</div>
	{/if}
	<div class="pointer-events-none absolute right-2 bottom-2 rounded-sm bg-chrome/75 px-1.5 py-1 font-mono text-2xs text-ink-dim">
		<span class="text-axis-x">x</span> {Math.floor(viewer.position[2])}
		<span class="ml-1 text-axis-y">y</span> {Math.floor(viewer.position[1])}
		<span class="ml-1 text-axis-z">z</span> {Math.floor(viewer.position[0])}
	</div>
	<div class="pointer-events-none absolute bottom-8 left-2 text-2xs text-ink-faint">drag to orbit · Shift-drag to pan · right-click to navigate</div>
	<div class="pointer-events-none absolute bottom-2 left-2 text-2xs text-ink-faint">{#if detail > 1}Simplified preview · {detail}× coarser{:else}{levelName}{/if}</div>
	{#if error}<p class="absolute inset-x-3 top-1/2 -translate-y-1/2 text-center text-2xs text-danger">{error} {#if total > 0}<button type="button" class="ml-2 cursor-pointer underline" onclick={retryChunks}>Retry</button>{/if}</p>{/if}
	{#if meshErrors.labels || meshErrors.suggestions}<p class="absolute inset-x-3 top-1/2 -translate-y-1/2 text-center text-2xs text-warn">{meshErrors.labels || meshErrors.suggestions}</p>{/if}
</div>
