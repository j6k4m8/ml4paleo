<script lang="ts">
	import { onDestroy, onMount, untrack } from "svelte";
	import type { ProjectImage } from "#lib/types.ts";
	import { splitIntoDeltas } from "../labels/deltas";
	import { indexedDbStorage, OpQueue } from "../labels/opqueue.svelte";
	import { PlaneMask } from "../labels/raster";
	import { ChunkStore } from "./chunks";
	import { absolute, loadLevels } from "./image";
	import { actionFor, KEYMAP, MOUSE } from "./keymap";
	import { type LabelClass, LabelLayer } from "./labels";
	import { imageLoader, WorkerPool } from "./loader";
	import PlaneView from "./PlaneView.svelte";
	import { ViewerState } from "./state.svelte";
	import { aspectOf, type Level, type Plane, PLANES, type Vec3 } from "./tiles";

	let { image, projectId }: { image: ProjectImage; projectId: string } = $props();

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
	let classes: LabelClass[] = $state([]);
	let error = $state("");
	let pool: WorkerPool | undefined;
	let hovered: Plane = PLANES.xy;
	let notice = $state("");
	const queue = new OpQueue(project, indexedDbStorage(project));
	const sizes = new Map<string, [number, number]>();
	const controller = new AbortController();

	const voxelSize = manifest.voxel_size_zyx;
	const unit = manifest.unit ?? "";

	onMount(async () => {
		try {
			levels = await loadLevels(zarrUrl, controller.signal);
			pool = new WorkerPool();
			images = new ChunkStore(imageLoader(pool, absolute(zarrUrl), levels), CACHE_BYTES);
			const layer = new LabelLayer(project, pool, viewer.shape);
			await layer.start();
			if (controller.signal.aborted) return layer.stop();
			layer.onStopped = () => (error = "Live label updates stopped. Reload the page to see others' edits.");
			labels = layer;
			classes = layer.classes;
			viewer.activeClass ??= classes[0]?.value ?? null;
			queue.onOutcome((outcome) => {
				if (outcome.op.kind !== "edit") return;
				if ("result" in outcome) {
					layer.settle(outcome.op.local, outcome.result.chunks);
				} else {
					layer.settle(outcome.op.local, null);
					notice = outcome.conflict
						? "Someone changed those labels while you drew; your polygon was dropped. Draw it again."
						: `That edit didn't save: ${outcome.error}`;
				}
			});
			// Edits a previous visit left unsent show until they're saved.
			for (const op of await queue.start()) if (op.kind === "edit") layer.applyLocal(op.local, op.deltas);
		} catch (e) {
			if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
		}
	});

	onDestroy(() => {
		controller.abort();
		queue.stop();
		labels?.stop();
		images?.keepOnly(new Set());
		pool?.close();
	});

	$effect(() => {
		void [viewer.opacity, viewer.showLabels, viewer.layout, viewer.brushRadius, viewer.protectLabels];
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
		const main = viewer.layout === "four" ? "xy" : viewer.layout;
		if (viewer.autoFit && plane.name === main) fit();
	}

	// --- editing -------------------------------------------------------------

	/** Send one edit of a plane mask, showing it at once. */
	function commit(plane: Plane, slice: number, mask: PlaneMask, value: number, onlyIf: string, tool: Record<string, unknown>, strict = false) {
		if (!labels) return;
		const volume = mask.toVolume(plane, slice);
		let baseVersions: Map<string, number> | undefined;
		if (strict) {
			// Strict edits need the version of every chunk they touch; without
			// them, the edit applies like a brush stroke.
			const keys = splitIntoDeltas(volume.mask, volume.shape, volume.origin, { value }).map((d) => d.key.join("/"));
			const known = keys.map((id) => [id, labels!.versionOf(id)] as const);
			if (known.every(([, version]) => version !== undefined)) baseVersions = new Map(known as [string, number][]);
			else strict = false;
		}
		const deltas = splitIntoDeltas(volume.mask, volume.shape, volume.origin, { value, onlyIf, baseVersions });
		notice = "";
		for (const op of queue.edit(deltas, { strict, tool })) labels.applyLocal(op.local, op.deltas);
	}

	function stroke(plane: Plane, slice: number, mask: PlaneMask) {
		const erase = viewer.tool === "eraser";
		const value = erase ? 0 : viewer.activeClass;
		if (value === null) return;
		const onlyIf = erase ? "any" : viewer.protectLabels ? "unlabeled" : "any";
		commit(plane, slice, mask, value, onlyIf, {
			name: erase ? "eraser" : "brush",
			radius: viewer.brushRadius,
			plane: plane.name,
			slice,
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
		if (event.key === " " && !(event.target instanceof HTMLInputElement)) {
			viewer.panning = true;
			event.preventDefault();
			return;
		}
		const action = actionFor(event);
		if (!action) return;
		event.preventDefault();
		const step = event.shiftKey ? 10 : 1;
		switch (action) {
			case "navigate":
			case "brush":
			case "eraser":
			case "polygon":
				return setTool(action);
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
			case "help":
				viewer.help = !viewer.help;
				return;
		}
	}

	function voxel(axis: number): string {
		const index = Math.floor(viewer.position[axis]!);
		if (!voxelSize || !unit) return String(index);
		return `${index} (${(index * voxelSize[axis]!).toFixed(1)} ${unit})`;
	}
</script>

<svelte:window onkeydown={key} onkeyup={keyUp} onblur={() => (viewer.panning = false)} />

<div class="viewer layout-{viewer.layout}">
	<div class="views">
		{#if images && levels.length > 0}
			{#each shown as plane (plane.name)}
				<PlaneView
					{plane}
					{viewer}
					{levels}
					{images}
					{labels}
					onhover={(p) => (hovered = p)}
					onresize={resized}
					onstroke={stroke}
					onpolygon={() => closePolygon()}
				/>
			{/each}
		{/if}
		<aside class="panel">
			<p class="position">
				<span class="axis-x">x {voxel(2)}</span>
				<span class="axis-y">y {voxel(1)}</span>
				<span class="axis-z">z {voxel(0)}</span>
			</p>
			<label>
				Window
				<span class="pair">
					<input type="number" step="any" bind:value={viewer.window[0]} aria-label="Window low" />
					<input type="number" step="any" bind:value={viewer.window[1]} aria-label="Window high" />
				</span>
			</label>
			<label class="row"><input type="checkbox" bind:checked={viewer.showLabels} /> Labels</label>
			<label>
				Label opacity
				<input type="range" min="0" max="1" step="0.05" bind:value={viewer.opacity} />
			</label>
			<div class="tools" role="radiogroup" aria-label="Tool">
				{#each [["navigate", "Navigate", "n"], ["brush", "Brush", "b"], ["eraser", "Eraser", "e"], ["polygon", "Polygon", "p"]] as [tool, name, shortcut] (tool)}
					<button
						class:secondary={viewer.tool !== tool}
						role="radio"
						aria-checked={viewer.tool === tool}
						title="{name} ({shortcut})"
						onclick={() => setTool(tool as typeof viewer.tool)}>{name}</button
					>
				{/each}
			</div>
			{#if classes.length > 0}
				<fieldset class="classes">
					<legend>Class</legend>
					{#each classes as label, index (label.value)}
						<label class="row">
							<input type="radio" name="class" value={label.value} bind:group={viewer.activeClass} />
							<span class="swatch" style:background={label.color}></span>
							{label.name}
							{#if index < 9}<kbd>{index + 1}</kbd>{/if}
						</label>
					{/each}
				</fieldset>
			{:else if labels}
				<p class="muted">Add label classes in the project settings to start labeling.</p>
			{/if}
			<label>
				Brush radius: {viewer.brushRadius} voxels
				<input type="range" min="0.5" max="64" step="0.5" bind:value={viewer.brushRadius} />
			</label>
			<label class="row"><input type="checkbox" bind:checked={viewer.protectLabels} /> Paint only unlabeled voxels</label>
			<div class="row history">
				<button class="secondary" disabled={queue.undoable === 0} onclick={() => queue.undo()} title="Undo (Ctrl+Z)">Undo</button>
				<button class="secondary" disabled={queue.redoable === 0} onclick={() => queue.redo()} title="Redo (Ctrl+Shift+Z)">Redo</button>
				<span class="status" class:error={!!queue.error} role="status">{status}</span>
			</div>
			{#if notice}<p class="error" role="alert">{notice}</p>{/if}
			<p class="muted">Press <kbd>?</kbd> for keys.</p>
			{#if error}<p class="error" role="alert">{error}</p>{/if}
		</aside>
	</div>
</div>

{#if viewer.help}
	<div class="help" role="dialog" aria-modal="true" aria-label="Keys">
		<table>
			<tbody>
				{#each KEYMAP as binding (binding.action)}
					<tr><td>{#each binding.keys as k, i (k)}{#if i > 0}, {/if}<kbd>{k}</kbd>{/each}</td><td>{binding.label}</td></tr>
				{/each}
				{#each MOUSE as [what, does] (what)}
					<tr><td>{what}</td><td>{does}</td></tr>
				{/each}
			</tbody>
		</table>
		<!-- svelte-ignore a11y_autofocus -->
		<button class="secondary" autofocus onclick={() => (viewer.help = false)}>Close</button>
	</div>
{/if}

<style>
	.viewer {
		height: 100%;
		min-height: 0;
	}
	.views {
		display: grid;
		gap: 4px;
		height: 100%;
		grid-template-columns: 1fr 1fr;
		grid-template-rows: 1fr 1fr;
	}
	.layout-xy .views,
	.layout-xz .views,
	.layout-yz .views {
		grid-template-columns: 1fr 16rem;
		grid-template-rows: 1fr;
	}
	.panel {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		padding: 0.5rem;
		overflow: auto;
		border: 1px solid var(--line);
		font-size: 0.9rem;
	}
	.panel label {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}
	.panel label.row {
		flex-direction: row;
		align-items: center;
		gap: 0.4rem;
	}
	.pair {
		display: flex;
		gap: 0.3rem;
	}
	.pair input {
		width: 50%;
		min-width: 0;
	}
	.position {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		margin: 0;
		font-variant-numeric: tabular-nums;
	}
	.axis-x {
		color: #e5534b;
	}
	.axis-y {
		color: #57ab5a;
	}
	.axis-z {
		color: #539bf5;
	}
	.tools {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	.tools button {
		padding: 0.25rem 0.5rem;
	}
	.classes {
		border: 1px solid var(--line);
		border-radius: 4px;
		margin: 0;
		padding: 0.3rem 0.5rem;
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}
	.classes kbd {
		margin-left: auto;
	}
	.history {
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.history button {
		padding: 0.2rem 0.5rem;
	}
	.status {
		margin-left: auto;
		font-size: 0.85rem;
	}
	.swatch {
		width: 0.8rem;
		height: 0.8rem;
		border-radius: 2px;
	}
	.help {
		position: fixed;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		background: var(--panel);
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 1rem;
		max-width: calc(100vw - 2rem);
		z-index: 10;
	}
	.help td {
		padding: 0.2rem 0.75rem 0.2rem 0;
	}
	kbd {
		font-family: ui-monospace, monospace;
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 0 0.3rem;
	}
	.muted {
		margin: 0;
	}
	@media (max-width: 700px) {
		.views,
		.layout-xy .views,
		.layout-xz .views,
		.layout-yz .views {
			grid-template-columns: 1fr;
			grid-template-rows: repeat(auto-fill, minmax(16rem, 1fr));
		}
	}
</style>
