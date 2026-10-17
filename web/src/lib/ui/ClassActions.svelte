<script lang="ts">
	import Ellipsis from "@lucide/svelte/icons/ellipsis";
	import { api, message } from "#lib/api.ts";
	import { BACKGROUND_VALUE } from "#lib/viewer/background.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";
	import ClassActionDialog from "./ClassActionDialog.svelte";

	let { pid, label, onchanged }: {
		pid: string;
		label: LabelClass;
		onchanged: (mode: "reset" | "delete", label: LabelClass) => void;
	} = $props();
	let menuOpen = $state(false);
	let pending: { mode: "reset" | "delete"; label: LabelClass; project: string; id: string } | null = $state(null);
	let confirmation = $state("");
	let busy = $state(false);
	let error = $state("");

	function choose(mode: "reset" | "delete") {
		menuOpen = false;
		confirmation = "";
		error = "";
		pending = { mode, label: { ...label }, project: pid, id: crypto.randomUUID() };
	}

	async function confirm(event: SubmitEvent) {
		event.preventDefault();
		const action = pending;
		if (!action || busy || confirmation !== action.label.name) return;
		busy = true;
		error = "";
		try {
			await api(`/api/projects/${action.project}/labels/classes/${action.label.value}/${action.mode}`, {
				body: { confirmation, client_op_id: action.id },
			});
			pending = null;
			if (pid === action.project) onchanged(action.mode, action.label);
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}
</script>

<details class="relative" bind:open={menuOpen}>
	<summary class="btn cursor-pointer list-none p-1 [&::-webkit-details-marker]:hidden" aria-label="Actions for {label.name}"><Ellipsis size={14} /></summary>
	<div class="panel absolute right-0 z-20 flex w-72 flex-col p-1 shadow-lg">
		<button class="rounded-sm p-2 text-left text-danger hover:bg-field" onclick={() => choose("reset")}>Reset this class (wipe all paint)</button>
		<button class="rounded-sm p-2 text-left text-danger hover:bg-field disabled:cursor-not-allowed disabled:text-ink-faint" disabled={label.value === BACKGROUND_VALUE} onclick={() => choose("delete")}>Delete this class (remove this label from the project)</button>
		{#if label.value === BACKGROUND_VALUE}<p class="px-2 pb-1 text-2xs text-ink-dim">Background is built in and cannot be deleted.</p>{/if}
	</div>
</details>

{#if pending}
	<ClassActionDialog label={pending.label} mode={pending.mode} bind:confirmation {busy} {error} onconfirm={confirm} oncancel={() => { pending = null; }} />
{/if}
