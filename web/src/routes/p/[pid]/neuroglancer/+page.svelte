<script lang="ts">
	import ExternalLink from "@lucide/svelte/icons/external-link";
	import Orbit from "@lucide/svelte/icons/orbit";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { answerView, ownWindowHref, type NeuroglancerView } from "#lib/neuroglancer.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";

	const pid = $derived(page.params.pid ?? "");
	let projectName = $state("");
	let view: NeuroglancerView = $state({ kind: "loading" });
	let frame: HTMLIFrameElement | undefined = $state();

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "Neuroglancer" }]);
	});

	$effect(() => {
		const project = pid;
		const controller = new AbortController();
		projectName = "";
		view = { kind: "loading" };
		void load(project, controller.signal);
		return () => controller.abort();
	});

	async function load(project: string, signal: AbortSignal) {
		try {
			projectName = (await api<{ name: string }>(`/api/projects/${project}`, { signal })).name;
			view = answerView(await api<{ url: string | null }>(`/api/projects/${project}/neuroglancer`, { signal }));
		} catch (e) {
			if (signal.aborted) return;
			// The project answered, so a 404 now is its image that isn't there.
			view = e instanceof ApiError && e.status === 404 && projectName ? { kind: "no-image" } : { kind: "error", message: message(e) };
		}
	}

	/** What the frame shows now, which is on this site, so the page can read it. */
	function frameHref(): string | null {
		try {
			return frame?.contentWindow?.location.href ?? null;
		} catch {
			return null;
		}
	}

	/** Point the link at the view as it is now, so the new window opens where people have looked around to. */
	function showCurrentView(event: Event, began: string) {
		(event.currentTarget as HTMLAnchorElement).href = ownWindowHref(frameHref(), location.origin, began);
	}
</script>

<div class="flex h-full flex-col">
	<div class="shrink-0"><ProjectTabs {pid} /></div>

	{#if view.kind === "ready"}
		{@const began = view.url}
		<div class="flex h-7 shrink-0 items-center gap-3 border-b border-edge bg-panel px-3 text-2xs text-ink-dim">
			<span class="truncate">The image, labels, prediction and final segmentation, to look at in Neuroglancer. Paint in the annotator.</span>
			<a
				class="ml-auto flex shrink-0 items-center gap-1"
				href={began}
				target="_blank"
				rel="noopener"
				data-sveltekit-reload
				onpointerenter={(event) => showCurrentView(event, began)}
				onfocus={(event) => showCurrentView(event, began)}
				onclick={(event) => showCurrentView(event, began)}
			>
				Open in its own window <ExternalLink size={12} />
			</a>
		</div>
		<!-- Neuroglancer is this site's own page, so it reads the data as whoever is signed in. -->
		<iframe bind:this={frame} title="Neuroglancer" src={began} class="min-h-0 w-full flex-1 border-0 bg-black"></iframe>
	{:else}
		<div class="grid min-h-0 flex-1 place-items-center overflow-auto bg-pasteboard p-6">
			{#if view.kind === "loading"}
				<p class="muted">Loading…</p>
			{:else if view.kind === "unavailable"}
				<div class="panel grid max-w-md place-items-center gap-2 p-8 text-center">
					<Orbit size={28} class="text-ink-faint" />
					<p>This server doesn't have Neuroglancer.</p>
					<p class="muted text-2xs">
						Whoever runs it can set it up: the server image has a build of Neuroglancer, and <span class="font-mono">M4P_NEUROGLANCER_DIR</span> says where
						the server finds it.
					</p>
				</div>
			{:else if view.kind === "no-image"}
				<div class="panel grid max-w-md place-items-center gap-2 p-8 text-center">
					<Orbit size={28} class="text-ink-faint" />
					<p>This project has no image yet.</p>
					<p class="muted">Upload a scan on the overview, and it will show up here.</p>
					<a class="btn btn-primary mt-1 no-underline" href="/p/{pid}">Go to the overview</a>
				</div>
			{:else}
				<div class="panel grid max-w-md place-items-center gap-2 p-8 text-center">
					<p class="error" role="alert">{view.message}</p>
					<a href="/p/{pid}">Back to the project</a>
				</div>
			{/if}
		</div>
	{/if}
</div>
