<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import { page } from "$app/state";
	import { ApiError, api } from "#lib/api.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Labeled from "#lib/ui/Labeled.svelte";
	import LabelImport from "#lib/ui/LabelImport.svelte";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";
	import type { LabelClass } from "#lib/viewer/labels.ts";

	const pid = $derived(page.params.pid ?? "");
	let projectName = $state("");
	// Whether the project has an image to put labels on, once the page has asked.
	let hasImage = $state<boolean | null>(null);
	let classes: LabelClass[] = $state([]);
	let counts: Record<string, number> | null = $state(null);

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Labels" }]);
	});

	$effect(() => {
		api<{ name: string }>(`/api/projects/${pid}`).then((p) => (projectName = p.name), () => {});
		api(`/api/projects/${pid}/image`).then(
			() => (hasImage = true),
			(e: unknown) => (hasImage = e instanceof ApiError && e.status === 404 ? false : null),
		);
		void load();
	});

	/** What has been labeled; again once labels come in from a file. */
	async function load() {
		try {
			classes = await api<LabelClass[]>(`/api/projects/${pid}/labels/classes`);
			counts = await api<Record<string, number>>(`/api/projects/${pid}/labels/counts`);
		} catch {
			// The page still works for importing.
		}
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto grid max-w-6xl gap-4 p-6 lg:grid-cols-[1fr_20rem]">
	<div class="flex min-w-0 flex-col gap-4">
		<h1>Labels</h1>
		<section class="panel self-start">
			<h2 class="panel-title">What's labeled</h2>
			<div class="flex flex-col gap-3 p-3">
				<p class="max-w-prose text-ink-dim">
					A model learns from two kinds of labels: what you're looking for, and background, which is everything else.
					Paint both in the annotator, or import labels from a file.
				</p>
				{#if counts}<Labeled {classes} {counts} />{/if}
				<a class="btn btn-primary self-start no-underline" href="/p/{pid}/annotate"><Brush size={13} /> Open the annotator</a>
			</div>
		</section>
	</div>

	<div class="flex flex-col gap-4 lg:self-start">
		<LabelImport {pid} {hasImage} onimported={load} />
	</div>
</div>
