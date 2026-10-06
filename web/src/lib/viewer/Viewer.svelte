<script lang="ts">
	import { onDestroy, onMount, untrack } from "svelte";
	import type { ProjectImage } from "#lib/types.ts";
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
		} catch (e) {
			if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
		}
	});

	onDestroy(() => {
		controller.abort();
		labels?.stop();
		images?.keepOnly(new Set());
		pool?.close();
	});

	$effect(() => {
		void [viewer.opacity, viewer.showLabels, viewer.layout];
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
		const action = actionFor(event);
		if (!action) return;
		event.preventDefault();
		const step = event.shiftKey ? 10 : 1;
		switch (action) {
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

<svelte:window onkeydown={key} />

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
			{#if classes.length > 0}
				<ul class="classes">
					{#each classes as label (label.value)}
						<li><span class="swatch" style:background={label.color}></span>{label.name}</li>
					{/each}
				</ul>
			{/if}
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
	.classes {
		list-style: none;
		margin: 0;
		padding: 0;
	}
	.classes li {
		display: flex;
		align-items: center;
		gap: 0.4rem;
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
