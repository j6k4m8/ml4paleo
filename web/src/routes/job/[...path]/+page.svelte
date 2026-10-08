<script lang="ts">
	import History from "@lucide/svelte/icons/history";
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import { parseJobId } from "#lib/v1.ts";

	// A v1 job's old links: its page, and its annotate, annotations, models, and
	// download pages. Signed in, you can bring the job over into a new project
	// of yours (or open the one you made from it before).
	const id = $derived(parseJobId(`/job/${page.params.path ?? ""}`));
	let busy = $state(false);
	let error = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: `v1 job ${id ?? ""}` }]);
	});

	async function claim() {
		busy = true;
		error = "";
		try {
			const claimed = await api<{ project_id: string }>(`/api/v1-jobs/${id}/claim`, { method: "POST" });
			await goto(`/p/${claimed.project_id}`, { replace: true });
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="mx-auto max-w-md p-6">
	<section class="panel">
		<h1 class="panel-title flex items-center gap-1.5"><History size={13} /> ml4paleo v1 job {id ?? ""}</h1>
		<div class="flex flex-col gap-3 p-3">
			{#if !id}
				<p class="error" role="alert">That isn't a v1 job's link.</p>
			{:else}
				<p class="text-ink-dim">
					Importing copies this job's scan, its annotation samples (as labels), and its last
					finished segmentation (as the prediction) into a new project of yours. The first person to import a job gets
					it.
				</p>
				{#if error}<p class="error" role="alert">{error}</p>{/if}
			{/if}
			<div class="flex flex-wrap gap-2">
				{#if id}
					<button class="btn btn-primary" disabled={busy} onclick={claim}>
						{busy ? "Importing…" : "Import this job"}
					</button>
				{/if}
				<a class="btn hover:no-underline" href="/import">Import other v1 jobs</a>
			</div>
		</div>
	</section>
</div>
