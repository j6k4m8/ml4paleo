<script lang="ts">
	import { goto } from "$app/navigation";
	import { api, message } from "#lib/api.ts";
	import type { Project } from "#lib/types.ts";

	let projects: Project[] | null = $state(null);
	let name = $state("");
	let error = $state("");

	$effect(() => {
		api<Project[]>("/api/projects").then(
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

<h1>Projects</h1>
{#if projects === null}
	<p class="muted">Loading…</p>
{:else if projects.length === 0}
	<p class="muted">No projects yet.</p>
{:else}
	<ul>
		{#each projects as project (project.id)}
			<li><a href="/p/{project.id}">{project.name}</a></li>
		{/each}
	</ul>
{/if}

<h2>New project</h2>
<form class="stack" onsubmit={create}>
	<label>Name <input bind:value={name} maxlength="100" required /></label>
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	<button>Create</button>
</form>
