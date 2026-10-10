<script lang="ts">
	import Check from "@lucide/svelte/icons/check";
	import LoaderCircle from "@lucide/svelte/icons/loader-circle";
	import TriangleAlert from "@lucide/svelte/icons/triangle-alert";
	import X from "@lucide/svelte/icons/x";
	import { trainingNotifications } from "#lib/training-notifications.svelte.ts";
</script>

<div class="pointer-events-none fixed right-3 bottom-9 left-3 z-50 flex max-h-[45vh] flex-col gap-2 overflow-y-auto sm:left-auto sm:w-96"
	role="region" aria-label="Training notifications" aria-live="polite" aria-relevant="additions text">
	{#each trainingNotifications.notices as notice (notice.id)}
		<div class="pointer-events-auto flex items-start gap-2 rounded-sm border border-edge bg-panel p-3 shadow-lg shadow-black/40" aria-atomic="true">
			{#if notice.kind === "started"}<LoaderCircle size={16} class="mt-0.5 shrink-0 animate-spin text-accent" />
			{:else if notice.kind === "ready"}<Check size={16} class="mt-0.5 shrink-0 text-ok" />
			{:else}<TriangleAlert size={16} class="mt-0.5 shrink-0 text-warn" />{/if}
			<div class="flex min-w-0 flex-1 flex-col gap-1">
				<p class="break-words">{notice.message}</p>
				<a class="text-2xs" href="/p/{notice.project}/models#model-{notice.model}">{notice.kind === "started" ? notice.name : "Open Models"}</a>
			</div>
			<button class="btn btn-ghost shrink-0 p-0.5" aria-label="Dismiss notification for {notice.name}" onclick={() => trainingNotifications.dismiss(notice.id)}><X size={14} /></button>
		</div>
	{/each}
</div>
