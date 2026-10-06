<script lang="ts">
	import { untrack } from "svelte";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { describe, revert, type Roi, RoiList } from "#lib/rois.svelte.ts";
	import { drawThumbnail } from "#lib/thumbnail.ts";
	import type { ProjectImage } from "#lib/types.ts";
	import { loadLevels } from "#lib/viewer/image.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";
	import type { Level } from "#lib/viewer/tiles.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import SquareDashed from "@lucide/svelte/icons/square-dashed";
	import Trash from "@lucide/svelte/icons/trash";

	const pid = $derived(page.params.pid ?? "");
	let rois = $state<RoiList | null>(null);
	let image = $state<ProjectImage | null>(null);
	let levels: Level[] = $state([]);
	let colors = $state(new Map<number, string>());
	let filter = $state<"all" | Roi["status"]>("all");
	let error = $state("");
	let projectName = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "ROIs" }]);
	});

	$effect(() => {
		api<{ name: string }>(`/api/projects/${pid}`).then((p) => (projectName = p.name), () => {});
	});

	const shown = $derived(
		(rois?.items ?? []).filter((r) => filter === "all" || r.status === filter).toReversed(),
	);

	$effect(() => {
		const list = new RoiList(pid);
		rois = list;
		const stopRefreshing = list.keepFresh();
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
		return () => {
			controller.abort();
			stopRefreshing();
		};
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
			}).catch(() => {
				canvas.classList.add("failed");
				canvas.title = "No preview";
			});
		});
		observer.observe(canvas);
		return () => {
			observer.disconnect();
			controller.abort();
		};
	}

	async function remove(roi: Roi) {
		if (!confirm(`Delete this ${roi.kind} ROI? Its labels stay.`)) return;
		await rois?.remove(roi.id);
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
	<div class="flex flex-wrap items-center gap-3">
		<h1>ROIs</h1>
		<div class="ml-auto flex overflow-hidden rounded-sm border border-edge" role="radiogroup" aria-label="Show">
			{#each ["all", "open", "complete", "skipped"] as option (option)}
				<label
					class="flex cursor-pointer items-center gap-1.5 px-2.5 py-1 capitalize
						{filter === option ? 'bg-accent-fill text-white' : 'bg-panel text-ink-dim hover:bg-raised hover:text-ink'}"
				>
					<input class="sr-only" type="radio" name="filter" value={option} bind:group={filter} />
					{option}
					<span class="font-mono text-2xs">
						{option === "all" ? (rois?.items.length ?? 0) : (rois?.items.filter((r) => r.status === option).length ?? 0)}
					</span>
				</label>
			{/each}
		</div>
	</div>

	{#if rois?.loaded && rois.items.length === 0}
		<div class="panel grid place-items-center gap-2 p-10 text-center">
			<SquareDashed size={28} class="text-ink-faint" />
			<p class="muted">No ROIs yet. In the annotator, pick the ROI tool (<span class="kbd">R</span>) and drag a box.</p>
		</div>
	{/if}

	<ul class="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-3">
		{#each shown as roi (roi.id)}
			<li
				class="panel flex flex-col overflow-hidden border-t-2
					{roi.status === 'complete' ? 'border-t-ok' : roi.status === 'skipped' ? 'border-t-ink-faint' : 'border-t-warn'}"
			>
				{#key levels.length}
					<a href="/p/{pid}/annotate?roi={roi.id}" aria-label="Open the {describe(roi)} ROI in the annotator" class="block bg-black">
						<!-- Drawn once per card; a status change doesn't redraw it. -->
						<canvas
							class="block aspect-square w-full object-contain [image-rendering:pixelated] [&.failed]:opacity-30"
							{@attach (canvas) => untrack(() => thumbnail(canvas, roi))}
						></canvas>
					</a>
				{/key}
				<div class="flex flex-col gap-1.5 p-2">
					<div class="flex items-baseline justify-between">
						<span class="font-medium capitalize">{roi.kind}</span>
						<span class="font-mono text-2xs text-ink-dim">{describe(roi).slice(roi.kind.length + 1)}</span>
					</div>
					<div class="grid grid-cols-2 gap-1">
						<select
							class="field"
							aria-label="Status of the {describe(roi)} ROI"
							value={roi.status}
							onchange={async (e) => {
								const select = e.currentTarget;
								if (!(await rois?.update(roi.id, { status: select.value as Roi["status"] }))) revert(select, roi.status);
							}}
						>
							<option value="open">open</option>
							<option value="complete">complete</option>
							<option value="skipped">skipped</option>
						</select>
						<select
							class="field"
							aria-label="Split of the {describe(roi)} ROI"
							value={roi.split}
							onchange={async (e) => {
								const select = e.currentTarget;
								if (!(await rois?.update(roi.id, { split: select.value as Roi["split"] }))) revert(select, roi.split);
							}}
						>
							<option value="train">train</option>
							<option value="val">validation</option>
						</select>
					</div>
					<button class="btn btn-ghost btn-danger self-end" onclick={() => remove(roi)} aria-label="Delete the {describe(roi)} ROI">
						<Trash size={13} /> Delete
					</button>
				</div>
			</li>
		{/each}
	</ul>
	{#if rois?.error}<p class="error" role="alert">{rois.error}</p>{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}
</div>
