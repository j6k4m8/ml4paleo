<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import Check from "@lucide/svelte/icons/check";
	import CloudOff from "@lucide/svelte/icons/cloud-off";
	import Eraser from "@lucide/svelte/icons/eraser";
	import Eye from "@lucide/svelte/icons/eye";
	import EyeOff from "@lucide/svelte/icons/eye-off";
	import Hand from "@lucide/svelte/icons/hand";
	import ImageOff from "@lucide/svelte/icons/image-off";
	import Keyboard from "@lucide/svelte/icons/keyboard";
	import LayoutGrid from "@lucide/svelte/icons/layout-grid";
	import LoaderCircle from "@lucide/svelte/icons/loader-circle";
	import Lock from "@lucide/svelte/icons/lock";
	import Maximize2 from "@lucide/svelte/icons/maximize-2";
	import PanelRight from "@lucide/svelte/icons/panel-right";
	import Pentagon from "@lucide/svelte/icons/pentagon";
	import Redo2 from "@lucide/svelte/icons/redo-2";
	import SquareDashed from "@lucide/svelte/icons/square-dashed";
	import Undo2 from "@lucide/svelte/icons/undo-2";
	import X from "@lucide/svelte/icons/x";
	import { onDestroy, onMount, untrack } from "svelte";
	import Histogram from "#lib/ui/Histogram.svelte";
	import Panel from "#lib/ui/Panel.svelte";
	import ToolButton from "#lib/ui/ToolButton.svelte";
	import { ApiError, api } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { ProjectImage } from "#lib/types.ts";
	import { splitIntoDeltas } from "../labels/deltas";
	import { indexedDbStorage, OpQueue, type QueuedEdit } from "../labels/opqueue.svelte";
	import { PlaneMask } from "../labels/raster";
	import { describe, revert, type Roi, RoiList, roiBox, thinAxis } from "../rois.svelte";
	import { ChunkStore } from "./chunks";
	import { absolute, loadLevels } from "./image";
	import { actionFor, forFocused, KEYMAP, MOUSE } from "./keymap";
	import { type LabelClass, LabelLayer } from "./labels";
	import { imageLoader, labelLoader, WorkerPool } from "./loader";
	import PlaneView from "./PlaneView.svelte";
	import { LAYOUTS, type Stroke, ViewerState } from "./state.svelte";
	import { aspectOf, type Level, type Plane, PLANES, type Vec3 } from "./tiles";

	let {
		image,
		projectId,
		title = "Image",
		roi: startRoi = null,
	}: { image: ProjectImage; projectId: string; title?: string; roi?: string | null } = $props();

	const CACHE_BYTES = 512 * 1024 * 1024;

	// The page makes a new viewer for each image.
	const { manifest, zarr_url: zarrUrl } = untrack(() => image);
	const project = untrack(() => projectId);
	const [, nz, ny, nx] = manifest.shape_czyx;
	const viewer = new ViewerState([nz, ny, nx], aspectOf(manifest.voxel_size_zyx));
	viewer.window = [...manifest.window];
	viewer.position = [nz / 2, ny / 2, nx / 2];

	let levels: Level[] = $state([]);
	let images: ChunkStore | null = $state(null);
	let labels: LabelLayer | null = $state(null);
	let prediction: ChunkStore | null = $state(null);
	let predictionModel = $state("");
	let classes: LabelClass[] = $state([]);
	let error = $state("");
	let pool: WorkerPool | undefined;
	let hovered: Plane = PLANES.xy;
	let notice = $state("");
	let classesOpen = $state(true);
	// On narrow screens the dock floats over the views until closed.
	let dockOpen = $state(false);
	const queue = new OpQueue(project, indexedDbStorage(session.current?.user.id ?? "", project));
	// Strict edits compare against the chunk versions current when they're
	// sent, after this page's earlier edits have landed; if any version is
	// unknown, the edit applies like a brush stroke instead.
	queue.beforeSend = (op: QueuedEdit) => {
		if (!op.strict || !labels) return op;
		const versions = op.deltas.map((d) => labels!.versionOf(d.key.join("/")));
		if (versions.some((v) => v === undefined)) return { ...op, strict: false };
		return { ...op, deltas: op.deltas.map((d, i) => ({ ...d, base_version: versions[i]! })) };
	};
	const rois = new RoiList(project);
	const firstRoi = untrack(() => startRoi);
	const sizes = new Map<string, [number, number]>();
	const controller = new AbortController();

	const voxelSize = manifest.voxel_size_zyx;
	const unit = manifest.unit ?? "";

	onMount(async () => {
		try {
			levels = await loadLevels(zarrUrl, controller.signal);
			rois.load().then(() => {
				const found = rois.items.find((r) => r.id === firstRoi);
				if (found) goTo(found);
				else if (firstRoi) rois.error = "That ROI isn't in this project any more.";
			});
			pool = new WorkerPool();
			images = new ChunkStore(imageLoader(pool, absolute(zarrUrl), levels), CACHE_BYTES);
			api<{ zarr_url: string; model_name: string | null }>(`/api/projects/${project}/prediction`).then(
				(found) => {
					if (!pool || controller.signal.aborted) return;
					// Predictions never change once made, so their chunks cache like the image's.
					prediction = new ChunkStore(labelLoader(pool, absolute(found.zarr_url), viewer.shape), 128 * 1024 * 1024, 4);
					predictionModel = found.model_name ?? "a model";
				},
				(e: unknown) => {
					if (!(e instanceof ApiError && e.status === 404)) error = e instanceof Error ? e.message : String(e);
				},
			);
			const layer = new LabelLayer(project, pool, viewer.shape);
			await layer.start();
			if (controller.signal.aborted) return layer.stop();
			layer.onStopped = () => (error = "Live label updates stopped. Reload the page to see others' edits.");
			labels = layer;
			classes = layer.classes;
			viewer.activeClass ??= classes[0]?.value ?? null;
			queue.onOutcome((outcome) => {
				if ("cancelled" in outcome) {
					layer.settle(outcome.op.local, null);
				} else if (outcome.op.kind !== "edit") {
					if ("result" in outcome) layer.noteVersions(outcome.result.chunks);
					else if (!outcome.alreadyDone) notice = `That ${outcome.op.kind} didn't go through: ${outcome.error}`;
				} else if ("result" in outcome) {
					layer.settle(outcome.op.local, outcome.result.chunks);
				} else {
					layer.settle(outcome.op.local, null);
					notice = outcome.conflict
						? "Someone changed those labels while you drew; your polygon was dropped. Draw it again."
						: `That edit didn't save: ${outcome.error}`;
				}
			});
			queue.onRequeue((ops) => {
				for (const op of ops) layer.applyLocal(op.local, op.deltas);
			});
			// Edits a previous visit left unsent show until they're saved.
			for (const op of await queue.start()) if (op.kind === "edit") layer.applyLocal(op.local, op.deltas);
		} catch (e) {
			if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
		}
	});

	const stopRefreshing = rois.keepFresh();

	onDestroy(() => {
		controller.abort();
		stopRefreshing();
		queue.stop();
		labels?.stop();
		images?.keepOnly(new Set());
		pool?.close();
	});

	$effect(() => {
		void [
			viewer.opacity,
			viewer.showLabels,
			viewer.layout,
			viewer.brushRadius,
			viewer.protectLabels,
			viewer.roiDepth,
			viewer.showPrediction,
			viewer.predictionOpacity,
		];
		viewer.savePreferences();
	});

	const shown = $derived(
		viewer.layout === "four" ? [PLANES.xy, PLANES.yz, PLANES.xz] : [PLANES[viewer.layout]],
	);

	function fit() {
		viewer.autoFit = true;
		const plane = viewer.layout === "four" ? PLANES.xy : PLANES[viewer.layout];
		const size = sizes.get(plane.name);
		if (!size) return;
		const { shape, aspect } = viewer;
		viewer.zoom = Math.min(size[0] / (shape[plane.u] * aspect[plane.u]), size[1] / (shape[plane.v] * aspect[plane.v]));
		const point = [...viewer.position] as Vec3;
		point[plane.u] = shape[plane.u] / 2;
		point[plane.v] = shape[plane.v] / 2;
		viewer.moveTo(point);
	}

	function resized(plane: Plane, width: number, height: number) {
		sizes.set(plane.name, [width, height]);
		if (pendingGoTo) return goTo(pendingGoTo);
		const main = viewer.layout === "four" ? "xy" : viewer.layout;
		if (plane.name === main && viewer.autoFit) fit();
	}

	// --- editing -------------------------------------------------------------

	/** Send one edit of a plane mask, showing it at once. */
	function commit(plane: Plane, slice: number, mask: PlaneMask, value: number, onlyIf: string, tool: Record<string, unknown>, strict = false) {
		if (!labels) return;
		const volume = mask.toVolume(plane, slice);
		const deltas = splitIntoDeltas(volume.mask, volume.shape, volume.origin, { value, onlyIf });
		notice = "";
		for (const op of queue.edit(deltas, { strict, tool })) labels.applyLocal(op.local, op.deltas);
	}

	function stroke(drawn: Stroke) {
		if (!drawn.erase && drawn.value === 0) return;
		commit(drawn.plane, drawn.slice, drawn.mask, drawn.value, drawn.onlyIf, {
			name: drawn.erase ? "eraser" : "brush",
			radius: drawn.radius,
			plane: drawn.plane.name,
			slice: drawn.slice,
		});
	}

	/** Fill the polygon being drawn; `erase` clears the active class inside it instead. */
	function closePolygon(erase = false) {
		const polygon = viewer.polygon;
		viewer.polygon = null;
		if (!polygon || polygon.points.length < 3 || viewer.activeClass === null) return;
		const plane = PLANES[polygon.plane];
		const mask = new PlaneMask(viewer.shape[plane.u], viewer.shape[plane.v]);
		mask.polygon(polygon.points);
		if (mask.count === 0) return;
		const value = erase ? 0 : viewer.activeClass;
		const onlyIf = erase ? `class:${viewer.activeClass}` : viewer.protectLabels ? "unlabeled" : "any";
		const points = polygon.points.map(([u, v]) => [Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
		const tool = { name: erase ? "polygon-erase" : "polygon", plane: plane.name, slice: polygon.slice, points: points.length <= 500 ? points : undefined };
		commit(plane, polygon.slice, mask, value, onlyIf, tool, true);
	}

	// --- ROIs ----------------------------------------------------------------

	async function drawRoi(plane: Plane, slice: number, corners: [[number, number], [number, number]]) {
		const bbox = roiBox(plane, slice, corners, viewer.roiDepth, viewer.shape);
		if (!bbox) return;
		const roi = await rois.add(bbox, viewer.roiDepth === 1 ? "slice" : "cube");
		if (roi) viewer.selectedRoi = roi.id;
	}

	// An ROI to fit once the main view knows its size.
	let pendingGoTo: Roi | null = null;

	/**
	 * Center the views on an ROI and fit it: a slice ROI in the view of its
	 * plane, a cube in every view shown.
	 */
	function goTo(roi: Roi) {
		viewer.selectedRoi = roi.id;
		viewer.autoFit = false;
		const { bbox } = roi;
		const thin = thinAxis(bbox);
		const slicePlane = thin === 0 ? PLANES.xy : thin === 1 ? PLANES.xz : PLANES.yz;
		viewer.moveTo([0, 1, 2].map((a) => (bbox[a]! + bbox[a + 3]!) / 2) as Vec3);
		if (roi.kind === "slice" && viewer.layout !== "four" && viewer.layout !== slicePlane.name) {
			// The new view's size arrives when it lays out; fit then.
			viewer.layout = slicePlane.name;
			sizes.clear();
			pendingGoTo = roi;
			return;
		}
		const planes = roi.kind === "slice" ? [slicePlane] : shown;
		const extent = (axis: number) => (bbox[axis + 3]! - bbox[axis]!) * viewer.aspect[axis]!;
		const zooms = planes.flatMap((plane) => {
			const size = sizes.get(plane.name);
			return size ? [Math.min(size[0] / extent(plane.u), size[1] / extent(plane.v))] : [];
		});
		pendingGoTo = zooms.length === planes.length ? null : roi;
		if (zooms.length > 0) viewer.zoom = Math.min(64, Math.max(1 / 512, 0.85 * Math.min(...zooms)));
	}

	const openRois = $derived(rois.items.filter((r) => r.status === "open"));

	/** The next open ROI after the selected one, in list order. */
	function nextOpen() {
		const items = rois.items;
		const at = items.findIndex((r) => r.id === viewer.selectedRoi);
		for (let step = 1; step <= items.length; step++) {
			const roi = items[(at + step) % items.length]!;
			if (roi.status === "open") return goTo(roi);
		}
	}

	function setTool(tool: typeof viewer.tool) {
		viewer.tool = tool;
		if (tool !== "polygon") viewer.polygon = null;
	}

	const status = $derived(
		queue.error
			? queue.error
			: queue.offline
				? `Offline · ${queue.pending} waiting`
				: queue.pending > 0
					? `Saving ${queue.pending}`
					: "Saved",
	);

	function keyUp(event: KeyboardEvent) {
		if (event.key === " ") viewer.panning = false;
	}

	/** The view that slice keys step: the only one, or the one last pointed at. */
	function activePlane(): Plane {
		return viewer.layout === "four" ? hovered : PLANES[viewer.layout];
	}

	function key(event: KeyboardEvent) {
		if (viewer.help && event.key === "Escape") {
			viewer.help = false;
			event.preventDefault();
			return;
		}
		if (event.key === " " && !forFocused(event)) {
			viewer.panning = true;
			event.preventDefault();
			return;
		}
		const action = actionFor(event);
		if (!action) return;
		// Enter and Backspace only mean something while drawing a polygon.
		if ((action === "close-polygon" || action === "remove-point") && !viewer.polygon) return;
		event.preventDefault();
		const step = event.shiftKey ? 10 : 1;
		switch (action) {
			case "navigate":
			case "brush":
			case "eraser":
			case "polygon":
			case "roi":
				return setTool(action);
			case "next-roi":
				return nextOpen();
			case "complete-roi":
				if (viewer.selectedRoi) rois.update(viewer.selectedRoi, { status: event.shiftKey ? "open" : "complete" });
				return;
			case "smaller":
				viewer.brushRadius = Math.max(0.5, Math.round(viewer.brushRadius / 1.25 * 2) / 2);
				return;
			case "bigger":
				viewer.brushRadius = Math.min(64, Math.max(viewer.brushRadius + 0.5, Math.round(viewer.brushRadius * 1.25 * 2) / 2));
				return;
			case "class": {
				const chosen = classes[Number(event.key) - 1];
				if (chosen) viewer.activeClass = chosen.value;
				return;
			}
			case "close-polygon":
				return closePolygon(event.altKey);
			case "remove-point":
				if (viewer.polygon) viewer.polygon = { ...viewer.polygon, points: viewer.polygon.points.slice(0, -1) };
				return;
			case "cancel":
				if (viewer.polygon) viewer.polygon = null;
				else setTool("navigate");
				return;
			case "undo":
				queue.undo();
				return;
			case "redo":
				queue.redo();
				return;
			case "slice-next":
				return viewer.step(activePlane().normal, step);
			case "slice-previous":
				return viewer.step(activePlane().normal, -step);
			case "zoom-in":
				viewer.autoFit = false;
				return viewer.zoomBy(1.25);
			case "zoom-out":
				viewer.autoFit = false;
				return viewer.zoomBy(0.8);
			case "fit":
				return fit();
			case "layout":
				viewer.nextLayout();
				return;
			case "labels":
				viewer.showLabels = !viewer.showLabels;
				return;
			case "prediction":
				viewer.showPrediction = !viewer.showPrediction;
				return;
			case "help":
				viewer.help = !viewer.help;
				return;
		}
	}

	/** Focus the keys dialog while it's open, and give focus back to whatever had it. */
	function holdFocus(dialog: HTMLElement) {
		const opener = document.activeElement;
		dialog.focus();
		return () => {
			if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
		};
	}

	function helpKey(event: KeyboardEvent) {
		// The viewer's keys wait until the dialog closes.
		event.stopPropagation();
		if (event.key === "Escape" || event.key === "?") {
			event.preventDefault();
			viewer.help = false;
		} else if (event.key === "Tab") {
			// Tab cycles through the dialog's controls without leaving it.
			event.preventDefault();
			const dialog = event.currentTarget as HTMLElement;
			const stops = [...dialog.querySelectorAll<HTMLElement>("a[href], button, input, select, textarea")];
			const at = stops.indexOf(document.activeElement as HTMLElement);
			stops.at(event.shiftKey ? (at <= 0 ? -1 : at - 1) : (at + 1) % stops.length)?.focus();
		}
	}

	function voxel(axis: number): string {
		const index = Math.floor(viewer.position[axis]!);
		if (!voxelSize || !unit) return String(index);
		return `${index} (${(index * voxelSize[axis]!).toFixed(1)} ${unit})`;
	}

	const TOOLS = [
		{ tool: "navigate", label: "Navigate", shortcut: "N", icon: Hand },
		{ tool: "brush", label: "Brush", shortcut: "B", icon: Brush },
		{ tool: "eraser", label: "Eraser", shortcut: "E", icon: Eraser },
		{ tool: "polygon", label: "Polygon", shortcut: "P", icon: Pentagon },
		{ tool: "roi", label: "ROI", shortcut: "R", icon: SquareDashed },
	] as const;

	const LAYOUT_NAMES = { four: "Four views", xy: "XY", xz: "XZ", yz: "YZ" } as const;

	const activeClass = $derived(classes.find((c) => c.value === viewer.activeClass));
	const zoomPercent = $derived(Math.round((viewer.zoom / (globalThis.devicePixelRatio || 1)) * 100));
	const [, imageZ, imageY, imageX] = manifest.shape_czyx;
	const histogram = manifest.histogram && !Array.isArray(manifest.histogram) ? manifest.histogram : null;
	const saveState = $derived(queue.error ? "error" : queue.offline ? "offline" : queue.pending > 0 ? "saving" : "saved");

	const hint = $derived(
		{
			navigate: "Drag to pan · wheel steps slices · Ctrl+wheel zooms · click moves the crosshair",
			brush: "Drag to paint the active class · [ ] change the size",
			eraser: "Drag to erase labels · [ ] change the size",
			polygon: "Click to add points · Enter or double-click fills · Alt+Enter erases the class inside · Esc cancels",
			roi: "Drag a box on a slice · G goes to the next open ROI · C marks it complete",
		}[viewer.tool],
	);
</script>

<svelte:window onkeydown={key} onkeyup={keyUp} onblur={() => (viewer.panning = false)} />

{#snippet problem(text: string, dismiss: () => void)}
	<div
		class="pointer-events-auto flex w-full max-w-lg items-start gap-2 rounded-sm border border-danger/50 bg-panel px-2.5 py-1.5 text-danger shadow-lg shadow-black/40"
		role="alert"
	>
		<span class="flex-1">{text}</span>
		<button class="text-ink-dim hover:text-ink" aria-label="Dismiss" onclick={dismiss}><X size={14} /></button>
	</div>
{/snippet}

<div class="flex h-full flex-col bg-chrome text-ink">
	<!-- Options bar: the active tool's settings, scrolling sideways when they don't fit. -->
	<div class="flex h-9 shrink-0 items-center border-b border-edge bg-panel">
		<div class="flex min-w-0 flex-1 items-center gap-3 self-stretch overflow-x-auto px-3 whitespace-nowrap">
			<span class="flex shrink-0 items-center gap-1.5 font-medium">
				{#each TOOLS as entry (entry.tool)}
					{#if entry.tool === viewer.tool}<entry.icon size={14} class="text-ink-dim" />{entry.label}{/if}
				{/each}
			</span>
			<span class="h-4 w-px shrink-0 bg-line"></span>
			{#if viewer.tool === "brush" || viewer.tool === "eraser"}
				<label class="flex shrink-0 items-center gap-2 text-ink-dim">
					Size
					<input class="w-28" type="range" min="0.5" max="64" step="0.5" bind:value={viewer.brushRadius} />
					<input class="field w-14 font-mono" type="number" min="0.5" max="64" step="0.5" bind:value={viewer.brushRadius} aria-label="Brush radius" />
				</label>
			{/if}
			{#if viewer.tool === "brush" || viewer.tool === "polygon"}
				<label class="flex shrink-0 items-center gap-1.5 text-ink-dim">
					<input type="checkbox" bind:checked={viewer.protectLabels} /> Only unlabeled voxels
				</label>
			{/if}
			{#if viewer.tool === "roi"}
				<label class="flex shrink-0 items-center gap-2 text-ink-dim">
					Depth
					<input class="w-28" type="range" min="1" max="256" step="1" bind:value={viewer.roiDepth} />
					<span class="w-24 font-mono text-ink">{viewer.roiDepth === 1 ? "slice" : `${viewer.roiDepth} voxels`}</span>
				</label>
			{/if}
			{#if viewer.tool === "navigate"}
				<div class="flex shrink-0 overflow-hidden rounded-sm border border-edge" role="radiogroup" aria-label="Layout">
					{#each LAYOUTS as layout (layout)}
						<button
							role="radio"
							aria-checked={viewer.layout === layout}
							class="flex h-6 items-center gap-1 px-2 {viewer.layout === layout ? 'bg-accent text-white' : 'bg-raised text-ink-dim hover:bg-hover hover:text-ink'}"
							onclick={() => (viewer.layout = layout)}
						>
							{#if layout === "four"}<LayoutGrid size={12} />{/if}
							{LAYOUT_NAMES[layout]}
						</button>
					{/each}
				</div>
				<button class="btn shrink-0" onclick={fit} title="Fit the image (0)"><Maximize2 size={12} /> Fit</button>
			{/if}
			<span class="ml-auto hidden truncate text-2xs text-ink-faint lg:inline">{hint}</span>
		</div>
		<!-- Outside the scrolling part, so it's always in reach. -->
		<button
			class="btn btn-ghost mx-1.5 shrink-0 md:hidden"
			aria-label="Panels"
			aria-expanded={dockOpen}
			onclick={() => (dockOpen = !dockOpen)}
		>
			<PanelRight size={14} />
		</button>
	</div>

	<div class="relative flex min-h-0 flex-1">
		<!-- Tools -->
		<nav class="flex w-11 shrink-0 flex-col items-center gap-0.5 border-r border-edge bg-panel py-1.5" aria-label="Tools">
			{#each TOOLS as entry (entry.tool)}
				<ToolButton
					icon={entry.icon}
					label={entry.label}
					shortcut={entry.shortcut}
					active={viewer.tool === entry.tool}
					onclick={() => setTool(entry.tool)}
				/>
			{/each}
			<span class="my-1.5 h-px w-6 bg-line"></span>
			<!-- The class brushes and polygons paint, like a foreground color. -->
			<button
				class="size-7 rounded-sm border-2 border-ink/80 shadow-[0_0_0_1px_black]"
				style:background={activeClass?.color ?? "transparent"}
				title={activeClass ? `Painting ${activeClass.name} (1–9 to change)` : "No class to paint"}
				aria-label={activeClass ? `Active class: ${activeClass.name}` : "No active class"}
				onclick={() => (classesOpen = true)}
			></button>
			<span class="my-1.5 h-px w-6 bg-line"></span>
			<ToolButton icon={Undo2} label="Undo" shortcut="Ctrl+Z" disabled={queue.undoable === 0} onclick={() => queue.undo()} />
			<ToolButton icon={Redo2} label="Redo" shortcut="Ctrl+Shift+Z" disabled={queue.redoable === 0} onclick={() => queue.redo()} />
			<div class="mt-auto">
				<ToolButton icon={Keyboard} label="Keys" shortcut="?" active={viewer.help} onclick={() => (viewer.help = !viewer.help)} />
			</div>
		</nav>

		<!-- Document -->
		<div class="relative flex min-w-0 flex-1 flex-col">
			<div class="flex h-7 shrink-0 items-end border-b border-edge bg-chrome px-2">
				<div class="flex h-6 items-center gap-2 rounded-t-sm bg-pasteboard px-3 shadow-[inset_0_1px_0_var(--color-accent)]">
					<span class="font-medium">{title}</span>
					<span class="font-mono text-2xs text-ink-faint">{imageX}×{imageY}×{imageZ} · {manifest.dtype}</span>
				</div>
			</div>
			<!-- What went wrong, over the views, where it shows even with the dock closed. -->
			{#if error || notice || rois.error}
				<div class="pointer-events-none absolute inset-x-0 top-8 z-30 flex flex-col items-center gap-1 px-2">
					{#if error}{@render problem(error, () => (error = ""))}{/if}
					{#if notice}{@render problem(notice, () => (notice = ""))}{/if}
					{#if rois.error}{@render problem(rois.error, () => (rois.error = ""))}{/if}
				</div>
			{/if}
			<div
				class="grid min-h-0 flex-1 gap-px bg-edge
					{viewer.layout === 'four' ? 'grid-cols-2 grid-rows-2' : 'grid-cols-1 grid-rows-1'}"
			>
				{#if images && levels.length > 0}
					{#each shown as plane (plane.name)}
						<PlaneView
							{plane}
							{viewer}
							{levels}
							{images}
							{labels}
							{prediction}
							onhover={(p) => (hovered = p)}
							onresize={resized}
							onstroke={stroke}
							onpolygon={() => closePolygon()}
							rois={rois.items}
							onroi={drawRoi}
						/>
					{/each}
					{#if viewer.layout === "four"}
						<div class="flex flex-col justify-center gap-1 bg-pasteboard p-4 font-mono text-2xs text-ink-dim">
							<span><span class="text-axis-x">x</span> {voxel(2)}</span>
							<span><span class="text-axis-y">y</span> {voxel(1)}</span>
							<span><span class="text-axis-z">z</span> {voxel(0)}</span>
						</div>
					{/if}
				{:else}
					<div class="grid place-items-center bg-pasteboard text-ink-faint">
						{#if error}<ImageOff size={20} />{:else}<LoaderCircle size={20} class="animate-spin" />{/if}
					</div>
				{/if}
			</div>
		</div>

		<!-- Dock -->
		<aside
			class="{dockOpen ? 'flex' : 'hidden'} absolute inset-y-0 right-0 z-20 w-64 shrink-0 flex-col overflow-y-auto border-l border-edge bg-panel shadow-2xl shadow-black/50 md:static md:flex md:shadow-none"
			aria-label="Panels"
		>
			<Panel title="Info">
				<div class="grid grid-cols-[1rem_1fr] gap-x-1 gap-y-0.5 font-mono text-2xs">
					<span class="text-axis-x">X</span><span>{voxel(2)}</span>
					<span class="text-axis-y">Y</span><span>{voxel(1)}</span>
					<span class="text-axis-z">Z</span><span>{voxel(0)}</span>
				</div>
			</Panel>

			<Panel title="Levels">
				{#if histogram}
					<Histogram counts={histogram.counts} edges={histogram.edges} bind:window={viewer.window} />
				{/if}
				<div class="grid grid-cols-2 gap-2">
					<label class="label">Black <input class="field font-mono" type="number" step="any" bind:value={viewer.window[0]} /></label>
					<label class="label">White <input class="field font-mono" type="number" step="any" bind:value={viewer.window[1]} /></label>
				</div>
			</Panel>

			<Panel title="Layers">
				<ul class="-mx-2.5 -my-2.5 flex flex-col divide-y divide-edge">
					<li class="flex flex-col gap-1.5 px-2.5 py-2">
						<div class="flex items-center gap-2">
							<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showLabels ? 'Hide' : 'Show'} labels" title="Show or hide (V)" onclick={() => (viewer.showLabels = !viewer.showLabels)}>
								{#if viewer.showLabels}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
							</button>
							<span class="flex-1">Labels</span>
							<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.opacity * 100)}%</span>
						</div>
						<input type="range" min="0" max="1" step="0.05" bind:value={viewer.opacity} aria-label="Label opacity" />
					</li>
					{#if prediction}
						<li class="flex flex-col gap-1.5 px-2.5 py-2">
							<div class="flex items-center gap-2">
								<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showPrediction ? 'Hide' : 'Show'} prediction" title="Show or hide (M)" onclick={() => (viewer.showPrediction = !viewer.showPrediction)}>
									{#if viewer.showPrediction}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
								</button>
								<span class="flex-1 truncate">Prediction <span class="text-ink-faint">· {predictionModel}</span></span>
								<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.predictionOpacity * 100)}%</span>
							</div>
							<input type="range" min="0" max="1" step="0.05" bind:value={viewer.predictionOpacity} aria-label="Prediction opacity" />
						</li>
					{/if}
					<li class="flex items-center gap-2 px-2.5 py-2 text-ink-dim">
						<Eye size={14} class="opacity-40" />
						<span class="flex-1">Image</span>
						<Lock size={12} />
					</li>
				</ul>
			</Panel>

			<Panel title="Classes" bind:open={classesOpen}>
				{#if classes.length > 0}
					<ul class="-mx-2.5 -my-1 flex flex-col">
						{#each classes as label, index (label.value)}
							<li>
								<button
									class="flex w-full items-center gap-2 px-2.5 py-1 text-left {viewer.activeClass === label.value ? 'bg-accent-soft text-ink' : 'hover:bg-raised'}"
									onclick={() => (viewer.activeClass = label.value)}
									aria-pressed={viewer.activeClass === label.value}
								>
									<span class="size-3 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={label.color}></span>
									<span class="flex-1">{label.name}</span>
									{#if index < 9}<span class="kbd">{index + 1}</span>{/if}
								</button>
							</li>
						{/each}
					</ul>
				{:else if labels}
					<p class="text-ink-dim">Add label classes in the project settings to start labeling.</p>
				{/if}
			</Panel>

			<Panel title="ROIs · {openRois.length} open">
				<ul class="-mx-2.5 -my-1 flex max-h-52 flex-col overflow-y-auto">
					{#each rois.items as roi (roi.id)}
						<li class="flex items-center gap-1.5 px-2.5 py-0.5 {roi.id === viewer.selectedRoi ? 'bg-accent-soft' : 'hover:bg-raised'}">
							<button class="flex flex-1 items-center gap-1.5 truncate text-left" onclick={() => goTo(roi)}>
								<span class="size-1.5 shrink-0 rounded-full {roi.status === 'complete' ? 'bg-ok' : roi.status === 'skipped' ? 'bg-ink-faint' : 'bg-warn'}"></span>
								<span class="truncate">{describe(roi)}</span>
							</button>
							<select
								class="field !h-5 !w-20 !text-2xs"
								aria-label="Status of the {describe(roi)} ROI"
								value={roi.status}
								onchange={async (e) => {
									const select = e.currentTarget;
									if (!(await rois.update(roi.id, { status: select.value as Roi["status"] }))) revert(select, roi.status);
								}}
							>
								<option value="open">open</option>
								<option value="complete">complete</option>
								<option value="skipped">skipped</option>
							</select>
						</li>
					{:else}
						<li class="px-2.5 text-ink-dim">Draw one with the ROI tool (R).</li>
					{/each}
				</ul>
				<a href="/p/{project}/rois" class="self-start text-2xs">Open the ROI gallery</a>
			</Panel>
		</aside>
	</div>

	<!-- Status bar -->
	<footer class="flex h-6 shrink-0 items-center gap-4 border-t border-edge bg-chrome px-3 font-mono text-2xs text-ink-dim">
		<span title="Zoom">{zoomPercent}%</span>
		<span>
			<span class="text-axis-x">x</span>{Math.floor(viewer.position[2])}
			<span class="text-axis-y">y</span>{Math.floor(viewer.position[1])}
			<span class="text-axis-z">z</span>{Math.floor(viewer.position[0])}
		</span>
		<span class="hidden sm:inline">{LAYOUT_NAMES[viewer.layout]}</span>
		<span class="ml-auto flex items-center gap-1.5 font-sans {saveState === 'error' ? 'text-danger' : saveState === 'offline' ? 'text-warn' : ''}" role="status">
			{#if saveState === "saved"}<Check size={12} class="text-ok" />{:else if saveState === "saving"}<LoaderCircle size={12} class="animate-spin" />{:else}<CloudOff size={12} />{/if}
			{status}
		</span>
	</footer>
</div>

{#if viewer.help}
	<div class="fixed inset-0 z-40 grid place-items-center bg-black/50 p-4" role="presentation" onclick={() => (viewer.help = false)}>
		<div
			class="panel max-h-[80vh] w-full max-w-lg overflow-auto shadow-2xl shadow-black/60"
			role="dialog"
			aria-modal="true"
			aria-label="Keys"
			tabindex="-1"
			{@attach holdFocus}
			onclick={(e) => e.stopPropagation()}
			onkeydown={helpKey}
		>
			<div class="panel-title">Keyboard shortcuts</div>
			<table class="w-full">
				<tbody>
					{#each KEYMAP as binding (binding.action)}
						<tr class="border-b border-edge">
							<td class="px-3 py-1.5 whitespace-nowrap">
								{#each binding.keys as k (k)}<span class="kbd mr-1">{k.replace("mod+", "Ctrl+")}</span>{/each}
							</td>
							<td class="px-3 py-1.5 text-ink-dim">{binding.label}</td>
						</tr>
					{/each}
					{#each MOUSE as [what, does] (what)}
						<tr class="border-b border-edge">
							<td class="px-3 py-1.5">{what}</td>
							<td class="px-3 py-1.5 text-ink-dim">{does}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<div class="flex justify-end p-2">
				<button class="btn" onclick={() => (viewer.help = false)}>Close</button>
			</div>
		</div>
	</div>
{/if}
