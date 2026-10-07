<script lang="ts">
	import LogOut from "@lucide/svelte/icons/log-out";
	import Trash from "@lucide/svelte/icons/trash";
	import UserMinus from "@lucide/svelte/icons/user-minus";
	import UserPlus from "@lucide/svelte/icons/user-plus";
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { Member, Project } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";

	// The parts of the page a change can fail in.
	type Part = "name" | "members" | "leave" | "delete";

	const pid = $derived(page.params.pid ?? "");
	let project: Project | null = $state(null);
	let members: Member[] = $state([]);
	let name = $state("");
	let username = $state("");
	let renamed = $state(false);
	let error = $state("");
	// What went wrong with the last change, and where to say it.
	let failed: { part: Part; message: string } | null = $state(null);
	// Set while a change is out, so a double click doesn't send two.
	let busy = $state(false);

	const me = $derived(session.current?.user.id);
	const owned = $derived(members.some((m) => m.is_owner && m.user_id === me));
	// The owner first, then everyone else by name, as the server sends them.
	const listed = $derived(members.toSorted((a, b) => Number(b.is_owner) - Number(a.is_owner)));

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: project?.name ?? "…", href: `/p/${pid}` }, { label: "Settings" }]);
	});

	$effect(() => {
		if (pid) load();
	});

	async function load() {
		try {
			project = await api<Project>(`/api/projects/${pid}`);
			name = project.name;
			members = project.members ?? [];
		} catch (e) {
			error = message(e);
		}
	}

	/** Make a change, saying what went wrong in `part` of the page. */
	async function run(part: Part, change: () => Promise<void>) {
		busy = true;
		failed = null;
		renamed = false;
		try {
			await change();
		} catch (e) {
			failed = { part, message: message(e) };
		} finally {
			busy = false;
		}
	}

	function rename(event: SubmitEvent) {
		event.preventDefault();
		run("name", async () => {
			project = await api<Project>(`/api/projects/${pid}`, { method: "PATCH", body: { name } });
			name = project.name;
			members = project.members ?? members;
			renamed = true;
		});
	}

	/** Add or remove someone, then show who's in now, since others may have changed it too. */
	function changeMembers(change: () => Promise<void>) {
		run("members", async () => {
			try {
				await change();
			} finally {
				members = await api<Member[]>(`/api/projects/${pid}/members`).catch(() => members);
			}
		});
	}

	function addMember(event: SubmitEvent) {
		event.preventDefault();
		changeMembers(async () => {
			members = await api<Member[]>(`/api/projects/${pid}/members`, { body: { username } });
			username = "";
		});
	}

	function removeMember(member: Member) {
		if (!confirm(`Remove ${member.username} from this project? What they've done in it stays.`)) return;
		changeMembers(async () => {
			await api(`/api/projects/${pid}/members/${member.user_id}`, { method: "DELETE" });
			members = members.filter((m) => m.user_id !== member.user_id);
		});
	}

	// Leaving or deleting a project you can't see anymore has nothing left to do.
	const gone = (e: unknown) => {
		if (!(e instanceof ApiError && e.status === 404)) throw e;
	};

	function leave() {
		if (!project || !me) return;
		if (!confirm(`Leave "${project.name}"? You can't open it again until someone adds you back.`)) return;
		run("leave", async () => {
			await api(`/api/projects/${pid}/members/${me}`, { method: "DELETE" }).catch(gone);
			await goto("/projects");
		});
	}

	function deleteProject() {
		if (!project) return;
		const question = `Delete "${project.name}"? Anything running in it stops, and everything in it is deleted for everyone. This can't be undone.`;
		if (!confirm(question)) return;
		run("delete", async () => {
			await api(`/api/projects/${pid}`, { method: "DELETE" }).catch(gone);
			await goto("/projects");
		});
	}
</script>

{#snippet problem(part: Part)}
	{#if failed?.part === part}<p class="error" role="alert">{failed.message}</p>{/if}
{/snippet}

<ProjectTabs {pid} />

{#if project}
	<div class="mx-auto flex max-w-2xl flex-col gap-4 p-6">
		<h1>Settings</h1>

		<section class="panel">
			<h2 class="panel-title">Name</h2>
			<form class="flex flex-col gap-3 p-3" onsubmit={rename}>
				<div class="flex items-end gap-2">
					<label class="label flex-1">
						Project name
						<input class="field" bind:value={name} maxlength="100" required />
					</label>
					<button class="btn btn-primary" disabled={busy || !name.trim() || name.trim() === project.name}>Rename</button>
				</div>
				{@render problem("name")}
				{#if renamed && name.trim() === project.name}<p class="text-ok" role="status">Renamed.</p>{/if}
			</form>
		</section>

		<section class="panel">
			<h2 class="panel-title">Collaborators</h2>
			<div class="flex flex-col gap-3 p-3">
				<p class="text-ink-dim">Collaborators can do everything in the project except delete it.</p>
				<ul class="flex flex-col divide-y divide-edge">
					{#each listed as member (member.user_id)}
						<li class="flex h-8 items-center gap-2">
							<span class="font-medium">{member.username}</span>
							{#if member.is_owner}<span class="rounded-sm bg-field px-1.5 py-0.5 text-2xs text-ink-dim">owner</span>{/if}
							{#if member.user_id === me}<span class="rounded-sm bg-accent-soft px-1.5 py-0.5 text-2xs text-ink">you</span>{/if}
							{#if !member.is_owner && member.user_id !== me}
								<button
									class="btn btn-ghost btn-danger ml-auto"
									disabled={busy}
									onclick={() => removeMember(member)}
									title="Remove {member.username}"
									aria-label="Remove {member.username}"
								>
									<UserMinus size={13} />
								</button>
							{/if}
						</li>
					{/each}
				</ul>
				<form class="flex items-end gap-2" onsubmit={addMember}>
					<label class="label flex-1">
						Add someone by their username
						<input class="field" bind:value={username} maxlength="64" autocomplete="off" autocapitalize="none" spellcheck="false" required />
					</label>
					<button class="btn" disabled={busy || !username.trim()}><UserPlus size={13} /> Add</button>
				</form>
				{@render problem("members")}
			</div>
		</section>

		{#if owned}
			<section class="panel">
				<h2 class="panel-title">Delete</h2>
				<div class="flex flex-col gap-3 p-3">
					<p class="text-ink-dim">
						Deleting the project stops anything running in it and deletes its scan, labels, models, and results for everyone.
						This can't be undone.
					</p>
					{@render problem("delete")}
					<button class="btn btn-danger self-start" disabled={busy} onclick={deleteProject}><Trash size={13} /> Delete this project</button>
				</div>
			</section>
		{:else if me}
			<section class="panel">
				<h2 class="panel-title">Leave</h2>
				<div class="flex flex-col gap-3 p-3">
					<p class="text-ink-dim">You can't open the project again until someone adds you back. What you've done in it stays.</p>
					{@render problem("leave")}
					<button class="btn btn-danger self-start" disabled={busy} onclick={leave}><LogOut size={13} /> Leave this project</button>
				</div>
			</section>
		{/if}
	</div>
{/if}
{#if error}<p class="error p-6" role="alert">{error}</p>{/if}
