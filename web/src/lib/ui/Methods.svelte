<script lang="ts">
	import Copy from "@lucide/svelte/icons/copy";
	import ChevronRight from "@lucide/svelte/icons/chevron-right";
	import { PAPER_URL } from "#lib/publication.ts";
	import { METHOD_SECTIONS, composeMethods, methodsResult, type MethodsEntry, type MethodSectionKey } from "#lib/methods.ts";
	let { entries, pid, error = "" }: { entries: MethodsEntry[]; pid: string; error?: string } = $props();
	let selected = $state<MethodSectionKey[]>(METHOD_SECTIONS.map(([key]) => key));
	let copied = $state("");
	let copyError = $state("");
	const entry = $derived(methodsResult(entries));
	const composed = $derived(composeMethods(entry, selected));
	const names = { prediction: "Prediction", segmentation: "Final segmentation", meshes: "Meshes" };
	function toggle(key: MethodSectionKey, checked: boolean) {
		selected = checked ? [...selected, key] : selected.filter((item) => item !== key);
		copied = copyError = "";
	}
	async function copy() {
		if (!composed.paragraph) return;
		copyError = "";
		try {
			const paragraph = composed.paragraph;
			await navigator.clipboard.writeText(paragraph);
			copied = paragraph;
		} catch {
			copyError = "Select the paragraph to copy it; clipboard access isn't available.";
		}
	}
</script>

<details class="panel group self-start lg:col-span-2">
	<summary class="panel-title flex cursor-pointer list-none items-center gap-1"><ChevronRight size={13} class="group-open:rotate-90" />Methods</summary>
	<div class="flex flex-col gap-3 p-3">
		{#if error}<p class="error" role="alert">{error}</p>
		{:else if entry}
			<fieldset class="flex flex-wrap gap-x-4 gap-y-2">
				<legend class="mb-2 text-2xs text-ink-dim">Include in paragraph</legend>
				{#each METHOD_SECTIONS as [key, label] (key)}
					{@const available = entry.sections.some((section) => section.key === key)}
					<label class="flex items-center gap-1.5 {available ? '' : 'text-ink-faint'}">
						<input type="checkbox" checked={available && selected.includes(key)} disabled={!available} onchange={(event) => toggle(key, event.currentTarget.checked)} />{label}
					</label>
				{/each}
			</fieldset>
			<div class="flex flex-wrap items-center gap-2">
				<span class="text-2xs text-ink-faint">Based on {names[entry.source].toLowerCase()} saved {new Date(entry.committed_at).toLocaleString()}</span>
				<button class="btn ml-auto" onclick={copy} disabled={!composed.paragraph}><Copy size={13} />{copied && copied === composed.paragraph ? "Copied" : "Copy paragraph"}</button>
			</div>
			{#if composed.paragraph}
			<p class="select-text rounded-sm border border-edge bg-field p-3 text-sm leading-relaxed">
				{#each composed.paragraph.split(PAPER_URL) as part, index (index)}
					{#if index > 0}<a href={PAPER_URL} target="_blank" rel="noreferrer">{PAPER_URL}</a>{/if}{part}
				{/each}
			</p>
			{:else}<p class="text-ink-dim">Select a section with saved details to build the paragraph.</p>{/if}
			{#if entry.model_id}<a class="self-start text-2xs" href="/p/{pid}/models#model-{entry.model_id}">Source model</a>{/if}
			{#each composed.warnings as warning (warning)}<p class="text-2xs text-warn">{warning}</p>{/each}
			<p class="text-2xs text-ink-faint">Based on this result’s saved settings. Review and add scan acquisition details before publication.</p>
			{#if copyError}<p class="error" role="alert">{copyError}</p>{/if}
		{:else}
			<p class="text-ink-dim">Make a prediction to generate a Methods paragraph from its saved settings.</p>
		{/if}
	</div>
</details>
