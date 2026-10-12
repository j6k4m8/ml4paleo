<script module lang="ts">
	// At most two visible cards read image chunks at once. Each card reads its
	// three planes in sequence; leaving the page cancels pending work.
	let tail: Promise<unknown>[] = [Promise.resolve(), Promise.resolve()];
	let lane = 0;
	function enqueue(work: () => Promise<void>) {
		const index = lane++ % tail.length;
		tail[index] = tail[index]!.then(work, work);
	}
</script>

<script lang="ts">
	import Box from "@lucide/svelte/icons/box";
	import type { ProjectSummary } from "#lib/types.ts";
	import { drawImageSlice } from "#lib/thumbnail.ts";
	import { loadLevels } from "#lib/viewer/image.ts";
	import { PLANES } from "#lib/viewer/tiles.ts";

	let { image }: { image: ProjectSummary["image"] } = $props();
	let ready = $state(false);
	let failed = $state(false);
	const planes = [PLANES.xy, PLANES.yz, PLANES.xz];
	const shape = $derived(image?.manifest.shape_czyx.slice(1) ?? [1, 1, 1]);
	const spacing: [number, number, number] = $derived(image?.manifest.voxel_size_zyx ?? [1, 1, 1]);
	const thin = $derived(shape[0]! * spacing[0] < 0.02 * Math.max(shape[1]! * spacing[1], shape[2]! * spacing[2]));

	function preview(node: HTMLElement) {
		const scan = image;
		ready = false;
		failed = false;
		if (!scan) return;
		const controller = new AbortController();
		const observer = new IntersectionObserver((entries) => {
			if (!entries.some((entry) => entry.isIntersecting)) return;
			observer.disconnect();
			enqueue(async () => {
				try {
					controller.signal.throwIfAborted();
					const levels = await loadLevels(scan.zarr_url, controller.signal);
					const canvases = node.querySelectorAll("canvas");
					for (const [index, canvas] of canvases.entries()) {
						await drawImageSlice(canvas, { imageUrl: scan.zarr_url, levels,
							plane: planes[index]!, window: scan.manifest.window, target: 96, signal: controller.signal });
					}
					if (!controller.signal.aborted) ready = true;
				} catch {
					if (!controller.signal.aborted) failed = true;
				}
			});
		}, { rootMargin: "100px" });
		observer.observe(node);
		return () => { observer.disconnect(); controller.abort(); };
	}
</script>

<div class="relative flex h-40 items-center justify-center gap-1 bg-field p-2" {@attach preview}>
	{#if image}
		<div class="grid h-full w-full grid-cols-2 grid-rows-2 gap-1" class:opacity-0={!ready}>
			{#each (thin ? planes.slice(0, 1) : planes) as plane, index (index)}
				<div class="relative flex min-h-0 min-w-0 items-center justify-center" class:col-span-2={thin} class:row-span-2={thin}>
					<canvas class="block max-h-full max-w-full" style:width={`${(thin ? 140 : 70) * shape[plane.u]! * spacing[plane.u] / (shape[plane.v]! * spacing[plane.v])}px`} style:aspect-ratio={`${shape[plane.u]! * spacing[plane.u]} / ${shape[plane.v]! * spacing[plane.v]}`} aria-label={`${["XY", "YZ", "XZ"][index]} middle slice`}></canvas>
					<span class="absolute top-0 left-1 text-[9px] text-ink-faint">{["XY", "YZ", "XZ"][index]}</span>
				</div>
			{/each}
		</div>
		{#if !ready}<span class="absolute text-2xs text-ink-faint">{failed ? "Preview unavailable" : "Loading preview…"}</span>{/if}
	{:else}
		<div class="flex flex-col items-center gap-2 text-ink-faint"><Box size={28} strokeWidth={1.2} /><span class="text-2xs">No scan yet</span></div>
	{/if}
</div>
