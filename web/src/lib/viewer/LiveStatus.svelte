<script lang="ts">
	import Brain from "@lucide/svelte/icons/brain";
	import Check from "@lucide/svelte/icons/check";
	import Clock from "@lucide/svelte/icons/clock";
	import LoaderCircle from "@lucide/svelte/icons/loader-circle";
	import Pause from "@lucide/svelte/icons/pause";
	import Sparkles from "@lucide/svelte/icons/sparkles";
	import TriangleAlert from "@lucide/svelte/icons/triangle-alert";
	import { tooltip } from "#lib/ui/tooltip.ts";
	import type { LearningPhase, PredictionPhase } from "./live.svelte";

	let { learning, prediction, progress, queuedEdits = false, paused = false, held = false }: {
		learning: LearningPhase;
		prediction: PredictionPhase;
		progress: { ready: number; total: number };
		queuedEdits?: boolean;
		paused?: boolean;
		held?: boolean;
	} = $props();
	const learningLabels: Record<LearningPhase, string> = {
		loading: "Loading", learning: "Learning", queued: "Learning queued", ready: "Learned",
		manual: "Needs a model", error: "Learning blocked", waiting: "Waiting for labels",
	};
	const predictionLabels: Record<PredictionPhase, string> = {
		predicting: "Predicting", queued: "Predictions queued", ready: "Predictions ready",
		waiting: "Predictions waiting", retrying: "Predictions retrying",
	};
</script>

<div class="rounded-sm bg-field px-2 py-2 text-xs" role="status" aria-label="Suggestion progress">
	{#if paused || held}
		<div class="flex items-center gap-2"><Pause size={13} aria-hidden="true" /> {paused ? "Paused" : "On hold"}</div>
	{:else}
		<div class="flex items-center gap-2">
			<Brain size={13} class="shrink-0 text-ink-dim" aria-hidden="true" />
			<span class="flex-1">{learningLabels[learning]}</span>
			{#if queuedEdits}
				<span class="flex items-center gap-1 text-2xs text-ink-dim" use:tooltip={"Newer edits will be used next."}>
					<Clock size={11} aria-hidden="true" /> Edits queued
				</span>
			{/if}
			{#if learning === "learning" || learning === "loading"}
				<LoaderCircle size={12} class="shrink-0 animate-spin text-accent motion-reduce:animate-none" aria-hidden="true" />
			{:else if learning === "ready"}
				<Check size={12} class="text-ink-dim" aria-hidden="true" />
			{:else if learning === "error"}
				<TriangleAlert size={12} class="text-danger" aria-hidden="true" />
			{:else if learning === "queued"}
				<Clock size={12} class="text-ink-dim" aria-hidden="true" />
			{/if}
		</div>
		<div class="mt-2 flex items-center gap-2">
			<Sparkles size={13} class="shrink-0 text-ink-dim" aria-hidden="true" />
			<span class="flex-1">{predictionLabels[prediction]}</span>
			{#if progress.total > 0}
				<span class="font-mono text-2xs text-ink-dim" aria-label={`${progress.ready} of ${progress.total} areas ready`}>{progress.ready}/{progress.total}</span>
			{/if}
			{#if prediction === "predicting"}
				<LoaderCircle size={12} class="shrink-0 animate-spin text-accent motion-reduce:animate-none" aria-hidden="true" />
			{:else if prediction === "retrying"}
				<TriangleAlert size={12} class="text-danger" aria-hidden="true" />
			{:else if prediction === "ready"}
				<Check size={12} class="text-ink-dim" aria-hidden="true" />
			{/if}
		</div>
		{#if progress.total > 0}
			<div class="mt-2 h-1 overflow-hidden rounded-full bg-line" role="progressbar" aria-label="Suggestion areas ready"
				aria-valuenow={progress.ready} aria-valuemin={0} aria-valuemax={progress.total}>
				<div class="h-full bg-accent" style:width={`${100 * progress.ready / progress.total}%`}></div>
			</div>
		{/if}
	{/if}
</div>
