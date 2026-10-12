<script lang="ts">
	import ArrowRight from "@lucide/svelte/icons/arrow-right";
	import Layers from "@lucide/svelte/icons/layers";
	import Brush from "@lucide/svelte/icons/brush";
	import Box from "@lucide/svelte/icons/box";
	import { PAPER_URL } from "#lib/publication.ts";
</script>

<div class="mx-auto flex max-w-5xl flex-col gap-12 px-6 py-12 sm:px-10 sm:py-16">
	<section class="max-w-3xl">
		<h1 class="text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">From scans to structures</h1>
		<p class="mt-6 max-w-2xl text-base leading-relaxed text-ink-dim">Segment CT and synchrotron X-ray images in your browser. Teach a model with a few annotations, review its suggestions, and turn your segmentation into a 3D mesh.</p>
		<div class="mt-7 flex flex-wrap items-center gap-3">
			<a href="/signup" class="btn btn-primary !h-9 !px-4 hover:no-underline">Get started <ArrowRight size={15} /></a>
			<a href="/login" class="btn !h-9 !px-4 hover:no-underline">Sign in</a>
			<a href={PAPER_URL} class="ml-1" target="_blank" rel="noreferrer">Read the paper ↗</a>
		</div>
	</section>

	<section class="grid gap-3 sm:grid-cols-3" aria-label="From scan to mesh">
		<article class="panel p-5"><Layers size={23} class="mb-5 text-accent-hover" /><h2 class="!text-sm !normal-case !tracking-normal !text-ink">1. Bring your scan</h2><p class="mt-2 text-sm leading-relaxed text-ink-dim">Upload image stacks or DICOM slices. Explore your specimen in three planes and in 3D.</p></article>
		<article class="panel p-5"><Brush size={23} class="mb-5 text-accent-hover" /><h2 class="!text-sm !normal-case !tracking-normal !text-ink">2. Label and learn</h2><p class="mt-2 text-sm leading-relaxed text-ink-dim">Paint the structures you want to study and their background. Train a model, then accept or correct its suggestions.</p></article>
		<article class="panel p-5"><Box size={23} class="mb-5 text-accent-hover" /><h2 class="!text-sm !normal-case !tracking-normal !text-ink">3. Use your results</h2><p class="mt-2 text-sm leading-relaxed text-ink-dim">Export your segmentations and STL, OBJ or GLB meshes for scientific analysis, publication, visualization, and 3D printing.</p></article>
	</section>

	<section class="border-t border-line pt-7">
		<div class="max-w-3xl text-sm leading-relaxed text-ink-dim"><p>ml4paleo is a collaborative open-science project developing simple, powerful machine learning tools for paleontological research. Our code is open source under the Apache 2.0 license, and community contributions are welcome.</p><div class="mt-4 flex flex-wrap gap-5"><a href="/about">About the project and team</a><a href="https://github.com/j6k4m8/ml4paleo" target="_blank" rel="noreferrer">Explore the code ↗</a></div></div>
	</section>
</div>
