<script lang="ts">
	import type { LabelClass } from "#lib/viewer/labels.ts";

	let {
		label, mode, confirmation = $bindable(""), busy, error, onconfirm, oncancel,
	}: {
		label: LabelClass;
		mode: "reset" | "delete";
		confirmation?: string;
		busy: boolean;
		error: string;
		onconfirm: (event: SubmitEvent) => void;
		oncancel: () => void;
	} = $props();
</script>

<dialog
	class="panel fixed inset-0 m-auto w-[min(28rem,calc(100%-2rem))] p-0 text-ink backdrop:bg-black/60"
	aria-labelledby="class-action-title" aria-describedby="class-action-warning"
	{@attach (node) => { node.showModal(); return () => node.close(); }}
	oncancel={(event) => { event.preventDefault(); if (!busy) oncancel(); }}
>
	<h2 class="panel-title" id="class-action-title">{mode === "reset" ? "Reset" : "Delete"} {label.name}?</h2>
	<form class="flex flex-col gap-3 p-4" onsubmit={onconfirm}>
		<p id="class-action-warning">
			This will erase all saved annotations for <strong>{label.name}</strong> across the entire scan.
			{mode === "reset" ? "The class will remain available." : "The class will also be removed from the project."}
			<strong class="text-danger">This cannot be undone.</strong>
		</p>
		<label class="label">
			Type “{label.name}” to confirm
			<input class="field" bind:value={confirmation} disabled={busy} autocomplete="off" autocapitalize="none" spellcheck="false" required />
		</label>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		<div class="flex justify-end gap-2">
			<button class="btn" type="button" disabled={busy} onclick={oncancel}>Cancel</button>
			<button class="btn btn-danger" disabled={busy || confirmation !== label.name}>
				{busy ? (mode === "reset" ? "Resetting…" : "Deleting…") : (mode === "reset" ? "Reset this class" : "Delete this class")}
			</button>
		</div>
	</form>
</dialog>
