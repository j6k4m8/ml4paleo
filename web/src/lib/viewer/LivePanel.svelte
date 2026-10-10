<script lang="ts">
	import Sparkles from "@lucide/svelte/icons/sparkles";
	import LiveStatus from "./LiveStatus.svelte";
	import { learningHelp, learningSchedule } from "./live-copy";
	import type { ModelPlugin } from "./live";
	import type { LivePreview } from "./live.svelte";

	let { live, plugins, disabled, modelsHref, ontoggle, onshow }: {
		live: LivePreview | null;
		plugins: ModelPlugin[];
		disabled: boolean;
		modelsHref: string;
		ontoggle: () => void;
		onshow: () => void;
	} = $props();
</script>

<div class="flex flex-col gap-2 text-xs">
	<p class="text-ink-dim">{live?.plugin?.capabilities.display_name ?? "Loading…"}</p>
	<button class="btn w-full" class:btn-primary={live?.enabled} disabled={!live?.plugin || disabled} onclick={ontoggle}>
		<Sparkles size={13} /> {live?.enabled ? "Stop suggestions" : "Start suggestions"}
	</button>
	<p>Striped = unsaved. <strong class="font-medium">Accept (<kbd class="font-mono">I</kbd>)</strong> keeps what you choose.</p>
	{#if live?.plugin?.capabilities.learning === "manual"}<a class="text-accent hover:underline" href={modelsHref}>Teach from labels on Models</a>{/if}

	{#if live?.enabled}
		<LiveStatus learning={live.learningPhase} prediction={live.predictionPhase} progress={live.progress}
			queuedEdits={live.queuedEdits} paused={live.paused} held={disabled} />
		{#if live.paused}<button class="btn w-full" {disabled} onclick={onshow}>Show suggestions</button>{/if}
		{#if live.learningError}
			<p class="text-danger" role="alert">{learningHelp(live.learningError)}</p>
			{#if live.model}<p class="text-ink-dim">Earlier suggestions still work; new learning is blocked.</p>{/if}
			<div class="flex items-center gap-3">
				<button class="btn" {disabled} onclick={() => live?.retryLearning()}>Try again</button>
				<a class="text-accent hover:underline" href={modelsHref}>Open Models</a>
			</div>
		{/if}
	{/if}

	<details class="text-ink-dim">
		<summary class="cursor-pointer">Help</summary>
		<div class="mt-1.5 flex flex-col gap-2">
			<p>Paint examples of your classes and Background, then start suggestions. They appear near your view first. Keep painting while they update.</p>
			<p>{learningSchedule(live?.plugin?.capabilities.learning)}</p>
			<p>Accept uses a brush or polygon. It never overwrites saved labels, and you can undo it. Painted and accepted labels both help future suggestions.</p>
			<p>The eye hides suggestions and pauses updates. Stop also hides them. Work already started may finish, but never changes saved labels. Suggestions start off when you open an image.</p>
			<p>Decline hides suggestions without labeling them as Background. Restore shows declined suggestions again.</p>
		</div>
	</details>
	<details class="text-ink-dim">
		<summary class="cursor-pointer">{live?.error ? "Settings & error details" : "Settings"}</summary>
		<div class="mt-1.5 flex flex-col gap-2">
			<label for="live-backend">Suggestion method</label>
			<select id="live-backend" class="input w-full" value={live?.plugin?.name ?? ""} {disabled}
				onchange={(event) => { const plugin = plugins.find((p) => p.name === event.currentTarget.value); if (plugin) void live?.select(plugin); }}>
				{#each plugins as plugin}<option value={plugin.name}>{plugin.capabilities.display_name}</option>{/each}
			</select>
			{#if live?.plugin}<p>Model type: {live.plugin.capabilities.family}. Hardware: {live.plugin.devices.join(", ").toUpperCase()}.</p>{/if}
			<p>A saved model is what the app has learned from labels. Automatic learning may need room for two: the one in use and its replacement. It uses the method's default settings.</p>
			<p>Removing a model stops its work but does not erase your saved labels.</p>
			{#if live?.model}
				<p>Using: {live.model.name}</p>
				<p>Model ID: {live.model.id.slice(-6)}. Saved-label revision: {live.model.training_set.label_seq}.</p>
			{/if}
			{#if live?.learningError}<p class="break-words">Learning error: {live.learningError}</p>{/if}
			{#if live?.predictionError}<p class="break-words">Suggestion error: {live.predictionError}</p>{/if}
			<a class="text-accent hover:underline" href={modelsHref}>Manage models</a>
		</div>
	</details>
</div>
