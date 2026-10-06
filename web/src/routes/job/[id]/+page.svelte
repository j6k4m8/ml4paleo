<script lang="ts">
	import History from "@lucide/svelte/icons/history";
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import { parseJobId } from "#lib/v1.ts";

	// A v1 job's old link. Visiting it signed in brings the job over into a
	// new project of yours (or opens the one you made from it before).
	const id = $derived(parseJobId(page.params.id ?? ""));
	let error = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: `v1 job ${id ?? ""}` }]);
	});

	$effect(() => {
		// Signed-out visitors are sent to sign in first, then back here.
		if (!session.current) return;
		if (!id) {
			error = "That isn't a v1 job's link.";
			return;
		}
		error = "";
		api<{ project_id: string }>(`/api/v1-jobs/${id}/claim`, { method: "POST" }).then(
			(claimed) => goto(`/p/${claimed.project_id}`, { replace: true }),
			(e) => (error = message(e)),
		);
	});
</script>

<div class="mx-auto max-w-md p-6">
	<section class="panel">
		<h1 class="panel-title flex items-center gap-1.5"><History size={13} /> ml4paleo v1 job {id ?? ""}</h1>
		<div class="flex flex-col gap-3 p-3">
			{#if error}
				<p class="error" role="alert">{error}</p>
				<a class="btn self-start hover:no-underline" href="/import">Import other v1 jobs</a>
			{:else}
				<p class="text-ink-dim">Bringing this job over from the old ml4paleo…</p>
			{/if}
		</div>
	</section>
</div>
