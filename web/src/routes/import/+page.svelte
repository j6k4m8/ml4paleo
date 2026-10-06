<script lang="ts">
	import ArrowRight from "@lucide/svelte/icons/arrow-right";
	import History from "@lucide/svelte/icons/history";
	import { onMount } from "svelte";
	import { api, message } from "#lib/api.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import { type V1Job, parseJobId, rememberedJobs } from "#lib/v1.ts";

	interface Outcome {
		busy?: boolean;
		project_id?: string;
		error?: string;
	}

	let jobs: V1Job[] = $state([]);
	let outcomes: Record<string, Outcome> = $state({});
	let pasted = $state("");
	let pasteError = $state("");

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: "Import from v1" }]);
	});

	onMount(() => {
		try {
			jobs = rememberedJobs(window.localStorage);
		} catch {
			jobs = [];
		}
	});

	async function claim(job: V1Job) {
		outcomes[job.id] = { busy: true };
		try {
			const claimed = await api<{ project_id: string }>(`/api/v1-jobs/${job.id}/claim`, { method: "POST" });
			outcomes[job.id] = { project_id: claimed.project_id };
		} catch (e) {
			outcomes[job.id] = { error: message(e) };
		}
	}

	function add(event: SubmitEvent) {
		event.preventDefault();
		if (busy) return;
		const id = parseJobId(pasted);
		pasteError = id ? "" : "Paste a v1 job's id (six letters and digits) or its link.";
		if (!id) return;
		const job = jobs.find((j) => j.id === id) ?? { id, name: "" };
		if (!jobs.includes(job)) jobs = [job, ...jobs];
		pasted = "";
		claim(job);
	}

	// One at a time: the server counts each import as a miss until it finds
	// the job, so several at once could use up the misses allowed.
	const busy = $derived(Object.values(outcomes).some((o) => o.busy));
</script>

<div class="mx-auto flex max-w-3xl flex-col gap-4 p-6">
	<h1 class="flex items-center gap-2"><History size={16} /> Import from ml4paleo v1</h1>
	<p class="text-ink-dim">
		The old ml4paleo had no accounts: anyone with a job's link could open it. Importing a job copies its scan, your
		annotation samples (as labels in complete slice ROIs), and its last segmentation (as the prediction) into a new
		project of yours. The first person to import a job gets it.
	</p>

	<form class="panel flex items-end gap-2 p-3" onsubmit={add}>
		<label class="label flex-1">
			A job's id or link
			<input class="field font-mono" bind:value={pasted} placeholder="AB12CD or https://…/job/AB12CD" />
		</label>
		<button class="btn btn-primary" disabled={busy}>Import</button>
	</form>
	{#if pasteError}<p class="error" role="alert">{pasteError}</p>{/if}

	<section class="panel">
		<h2 class="panel-title">Jobs this browser opened</h2>
		{#if jobs.length === 0}
			<p class="p-3 text-ink-dim">
				This browser has no v1 jobs saved. Open a job's old link, or paste its id above.
			</p>
		{:else}
			<p class="px-3 pt-3 text-ink-dim">
				v1 saved every job a browser opened, so this list can include jobs other people shared with you. Import only
				your own.
			</p>
			<ul class="flex flex-col divide-y divide-edge">
				{#each jobs as job (job.id)}
					{@const outcome = outcomes[job.id]}
					<li class="flex flex-wrap items-center gap-2 px-3 py-2">
						<span class="font-mono text-2xs text-ink-dim">{job.id}</span>
						<span class="min-w-0 flex-1 truncate">{job.name || "Untitled"}</span>
						{#if outcome?.project_id}
							<a class="btn hover:no-underline" href="/p/{outcome.project_id}">Open <ArrowRight size={13} /></a>
						{:else}
							{#if outcome?.error}<span class="error text-2xs" role="alert">{outcome.error}</span>{/if}
							<button class="btn" disabled={busy} onclick={() => claim(job)}>
								{outcome?.busy ? "Importing…" : "Import"}
							</button>
						{/if}
					</li>
				{/each}
			</ul>
		{/if}
	</section>
</div>
