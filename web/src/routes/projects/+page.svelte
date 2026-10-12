<script lang="ts">
	import Box from "@lucide/svelte/icons/box";
	import Plus from "@lucide/svelte/icons/plus";
	import { goto } from "$app/navigation";
	import { api, message } from "#lib/api.ts";
	import type { Project, ProjectSummary } from "#lib/types.ts";
	import ProjectPreview from "#lib/ui/ProjectPreview.svelte";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";

	let projects: ProjectSummary[] | null = $state(null);
	let name = $state("");
	let error = $state("");
	let creating = $state(false);

	$effect(() => crumbs.set([{ label: "Projects" }]));

	$effect(() => {
		api<ProjectSummary[]>("/api/projects").then(
			(list) => (projects = list),
			(e) => (error = message(e)),
		);
	});

	async function create(event: SubmitEvent) {
		event.preventDefault();
		error = "";
		try {
			const project = await api<Project>("/api/projects", { body: { name } });
			goto(`/p/${project.id}`);
		} catch (e) {
			error = message(e);
		}
	}
</script>

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
	<div class="flex items-center gap-3">
		<h1>Projects</h1>
		<a class="btn btn-ghost ml-auto hover:no-underline" href="/import">Import from v1</a>
		<button class="btn btn-primary" onclick={() => (creating = !creating)}>
			<Plus size={14} /> New project
		</button>
	</div>

	{#if creating}
		<form class="panel flex items-end gap-2 p-3" onsubmit={create}>
			<label class="label flex-1">
				Name
				<!-- svelte-ignore a11y_autofocus -->
				<input class="field" bind:value={name} maxlength="100" placeholder="Skull, Ammonite 3…" required autofocus />
			</label>
			<button class="btn btn-primary">Create</button>
			<button type="button" class="btn btn-ghost" onclick={() => (creating = false)}>Cancel</button>
		</form>
	{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}

	{#if projects === null}
		<p class="muted">Loading…</p>
	{:else if projects.length === 0}
		<div class="panel grid place-items-center gap-2 p-10 text-center">
			<Box size={28} class="text-ink-faint" />
			<p class="muted">No projects yet. Make one, then upload a scan.</p>
		</div>
	{:else}
		<ul class="grid grid-cols-[repeat(auto-fill,minmax(13rem,1fr))] gap-3">
			{#each projects as project (project.id)}
				<li>
					<a
						href="/p/{project.id}"
						class="group panel flex flex-col overflow-hidden text-ink transition-colors hover:border-accent hover:no-underline"
					>
						<ProjectPreview image={project.image} />
						<div class="flex flex-col gap-0.5 border-t border-edge px-3 py-2">
							<span class="truncate font-medium">{project.name}</span>
							<span class="text-2xs text-ink-dim" title="Compressed scan size and saved, trained models (excluding Live checkpoints)">
								{project.image_bytes > 0 && project.image_bytes < 0.01 * 1024 ** 3 ? "<0.01" : (project.image_bytes / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 2 })} GB · {project.model_count} {project.model_count === 1 ? "model" : "models"}
							</span>
							<span class="text-2xs text-ink-faint">{new Date(project.created_at).toLocaleDateString()}</span>
						</div>
					</a>
				</li>
			{/each}
		</ul>
	{/if}
</div>
