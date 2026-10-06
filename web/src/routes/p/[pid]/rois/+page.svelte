<script lang="ts">
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { type Roi, RoiList } from "#lib/rois.svelte.ts";
	import { drawThumbnail } from "#lib/thumbnail.ts";
	import type { ProjectImage } from "#lib/types.ts";
	import { loadLevels } from "#lib/viewer/image.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";
	import type { Level } from "#lib/viewer/tiles.ts";

	const pid = $derived(page.params.pid ?? "");
	let rois = $state<RoiList | null>(null);
	let image = $state<ProjectImage | null>(null);
	let levels: Level[] = $state([]);
	let colors = $state(new Map<number, string>());
	let filter = $state<"all" | Roi["status"]>("all");
	let error = $state("");

	const shown = $derived(
		(rois?.items ?? []).filter((r) => filter === "all" || r.status === filter).toReversed(),
	);

	$effect(() => {
		const list = new RoiList(pid);
		rois = list;
		const controller = new AbortController();
		(async () => {
			try {
				await list.load();
				image = await api<ProjectImage>(`/api/projects/${pid}/image`);
				const classes = await api<LabelClass[]>(`/api/projects/${pid}/labels/classes`);
				colors = new Map(classes.map((c) => [c.value, c.color]));
				levels = await loadLevels(image.zarr_url, controller.signal);
			} catch (e) {
				if (!(e instanceof ApiError && e.status === 404)) error = message(e);
			}
		})();
		return () => controller.abort();
	});

	/** Draw a thumbnail once its card is on screen. */
	function thumbnail(canvas: HTMLCanvasElement, roi: Roi) {
		const controller = new AbortController();
		const observer = new IntersectionObserver(([entry]) => {
			if (!entry?.isIntersecting || !image || levels.length === 0) return;
			observer.disconnect();
			drawThumbnail(canvas, {
				imageUrl: image.zarr_url,
				labelsUrl: `/api/projects/${pid}/labels/zarr/`,
				levels,
				bbox: roi.bbox,
				window: image.manifest.window,
				colors,
				opacity: 0.5,
				signal: controller.signal,
			}).catch(() => canvas.classList.add("failed"));
		});
		observer.observe(canvas);
		return () => {
			observer.disconnect();
			controller.abort();
		};
	}

	function size(roi: Roi): string {
		const [z0, y0, x0, z1, y1, x1] = roi.bbox;
		return `${x1 - x0} × ${y1 - y0} × ${z1 - z0}`;
	}

	async function remove(roi: Roi) {
		if (!confirm(`Delete this ${roi.kind} ROI? Its labels stay.`)) return;
		await rois?.remove(roi.id);
	}
</script>

<p><a href="/p/{pid}">← Project</a> · <a href="/p/{pid}/annotate">Annotator</a></p>
<h1>ROIs</h1>

<div class="filters" role="radiogroup" aria-label="Show">
	{#each ["all", "open", "complete", "skipped"] as option (option)}
		<label>
			<input type="radio" name="filter" value={option} bind:group={filter} />
			{option}
			({option === "all" ? (rois?.items.length ?? 0) : (rois?.items.filter((r) => r.status === option).length ?? 0)})
		</label>
	{/each}
</div>

{#if rois && rois.items.length === 0}
	<p class="muted">No ROIs yet. In the annotator, pick the ROI tool (<kbd>r</kbd>) and drag a box.</p>
{/if}

<ul class="gallery">
	{#each shown as roi (roi.id)}
		<li class="card status-{roi.status}">
			{#key levels.length}
				<a href="/p/{pid}/annotate?roi={roi.id}" aria-label="Open this ROI in the annotator">
					<canvas {@attach (canvas) => thumbnail(canvas, roi)}></canvas>
				</a>
			{/key}
			<div class="meta">
				<strong>{roi.kind}</strong>
				<span class="muted">{size(roi)}</span>
			</div>
			<div class="actions">
				<select
					aria-label="Status"
					value={roi.status}
					onchange={(e) => rois?.update(roi.id, { status: e.currentTarget.value as Roi["status"] })}
				>
					<option value="open">open</option>
					<option value="complete">complete</option>
					<option value="skipped">skipped</option>
				</select>
				<select
					aria-label="Split"
					value={roi.split}
					onchange={(e) => rois?.update(roi.id, { split: e.currentTarget.value as Roi["split"] })}
				>
					<option value="train">train</option>
					<option value="val">validation</option>
				</select>
				<button class="secondary" onclick={() => remove(roi)}>Delete</button>
			</div>
		</li>
	{/each}
</ul>
{#if rois?.error}<p class="error" role="alert">{rois.error}</p>{/if}
{#if error}<p class="error" role="alert">{error}</p>{/if}

<style>
	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		margin-bottom: 1rem;
	}
	.gallery {
		list-style: none;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(12rem, 1fr));
		gap: 0.75rem;
	}
	.card {
		border: 1px solid var(--line);
		border-left: 4px solid var(--status);
		border-radius: 4px;
		padding: 0.5rem;
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		background: var(--panel);
	}
	.status-open {
		--status: #e3b341;
	}
	.status-complete {
		--status: #57ab5a;
	}
	.status-skipped {
		--status: #768390;
	}
	canvas {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		object-fit: contain;
		image-rendering: pixelated;
		background: #000;
	}
	.meta {
		display: flex;
		justify-content: space-between;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
	}
	.actions button {
		padding: 0.2rem 0.5rem;
	}
	kbd {
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 0 0.3rem;
	}
</style>
