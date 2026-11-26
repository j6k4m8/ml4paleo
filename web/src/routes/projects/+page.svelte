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
	<div class="flex flex-wrap items-center gap-x-3 gap-y-2">
		<h1>Projects</h1>
		<a class="btn btn-ghost ml-auto hover:no-underline" href="/import">Import from v1</a>
		{#if projects?.length !== 0}
			<button class="btn btn-primary" aria-expanded={creating} onclick={() => (creating = !creating)}>
				<Plus size={14} /> New project
			</button>
		{/if}
	</div>

	{#if creating}
		<form class="panel flex flex-wrap items-end gap-2 p-3" onsubmit={create}>
			<label class="label min-w-36 flex-1">
				Project name
				<!-- svelte-ignore a11y_autofocus -->
				<input class="field" bind:value={name} maxlength="100" placeholder="Skull, Ammonite 3…" required autofocus />
			</label>
			<div class="flex gap-2">
				<button class="btn btn-primary">Create project</button>
				<button type="button" class="btn btn-ghost" onclick={() => (creating = false)}>Cancel</button>
			</div>
		</form>
	{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}

	{#if projects === null}
		<p class="muted">Loading…</p>
	{:else if projects.length === 0}
		{#if !creating}
			<div class="panel grid place-items-center gap-3 p-10 text-center">
				<Box size={28} class="text-ink-faint" />
				<p class="muted">Create a project for the scan you want to segment.</p>
				<button class="btn btn-primary h-7" onclick={() => (creating = true)}>
					<Plus size={14} /> New project
				</button>
			</div>
		{/if}
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
							<span class="text-2xs text-ink-dim" title={project.image ? "Compressed scan size and saved, trained models (excluding Live checkpoints)" : undefined}>
								{#if project.image}
									{project.image_bytes > 0 && project.image_bytes < 0.01 * 1024 ** 3 ? "<0.01" : (project.image_bytes / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 2 })} GB · {project.model_count} {project.model_count === 1 ? "model" : "models"}
								{:else}
									Awaiting scan
								{/if}
							</span>
							<span class="text-2xs text-ink-faint">{new Date(project.created_at).toLocaleDateString()}</span>
						</div>
					</a>
				</li>
			{/each}
		</ul>
	{/if}
</div>
