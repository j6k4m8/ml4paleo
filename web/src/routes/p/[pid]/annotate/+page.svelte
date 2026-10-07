<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import type { Project, ProjectImage } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import { parseBox } from "#lib/viewer/link.ts";
	import Viewer from "#lib/viewer/Viewer.svelte";

	const pid = $derived(page.params.pid ?? "");
	// The image and the project it belongs to, set together.
	let shown: { pid: string; name: string; image: ProjectImage } | null = $state(null);
	let error = $state("");

	$effect(() => {
		crumbs.set([
			{ label: "Projects", href: "/projects" },
			{ label: shown?.name ?? "…", href: `/p/${pid}` },
			{ label: "Annotate" },
		]);
	});

	$effect(() => {
		const project = pid;
		const controller = new AbortController();
		shown = null;
		error = "";
		Promise.all([
			api<ProjectImage>(`/api/projects/${project}/image`, { signal: controller.signal }),
			api<Project>(`/api/projects/${project}`, { signal: controller.signal }),
		]).then(
			([image, info]) => (shown = { pid: project, name: info.name, image }),
			(e) => {
				if (!controller.signal.aborted) error = message(e);
			},
		);
		return () => controller.abort();
	});
</script>

{#if shown}
	{#key `${shown.pid}/${shown.image.artifact_id}`}
		<Viewer
			image={shown.image}
			projectId={shown.pid}
			title={shown.name}
			roi={page.url.searchParams.get("roi")}
			box={parseBox(page.url.searchParams.get("box"))}
		/>
	{/key}
{:else if error}
	<div class="grid h-full place-items-center bg-pasteboard">
		<div class="panel flex flex-col items-center gap-2 p-6">
			<p class="error" role="alert">{error}</p>
			<a href="/p/{pid}">Back to the project</a>
		</div>
	</div>
{/if}
