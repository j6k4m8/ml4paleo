<script lang="ts">
	import FileUp from "@lucide/svelte/icons/file-up";
	import X from "@lucide/svelte/icons/x";
	import { untrack } from "svelte";
	import { api, message } from "#lib/api.ts";
	import {
		BACKGROUND,
		defaultTargets,
		type LabelImport,
		labelFile,
		newClasses,
		nextColor,
		startBody,
		type Target,
	} from "#lib/labelimport.ts";
	import { unfinished } from "#lib/pipelines.ts";
	import type { Pipeline, Upload } from "#lib/types.ts";
	import { uploadFile } from "#lib/upload.ts";
	import type { LabelClass } from "#lib/viewer/labels.ts";

	let {
		pid,
		hasImage,
		onimported,
	}: {
		pid: string;
		/** Whether the project has an image to put labels on; null until known. */
		hasImage: boolean | null;
		onimported?: () => void;
	} = $props();

	let imports = $state<LabelImport[]>([]);
	// The import shown in full: one being checked, ready to import, or
	// importing, or the last one followed until the person moves on.
	let current = $state<LabelImport | null>(null);
	let classes = $state<LabelClass[]>([]);
	let targets = $state(new Map<number, Target>());
	let overwrite = $state(false);
	let uploading = $state<string | null>(null);
	let uploaded = $state(0);
	let busy = $state(false);
	let dragging = $state(false);
	let error = $state("");
	// Bumped to open the stream again when it breaks while the import runs.
	let reconnect = $state(0);

	const base = $derived(`/api/projects/${pid}/labels/imports`);
	const ACTIVE = ["checking", "ready", "importing"];
	const STATES: Record<LabelImport["state"], string> = {
		checking: "Checking",
		ready: "Checked",
		importing: "Importing",
		done: "Imported",
		failed: "Failed",
		cancelled: "Stopped",
		expired: "Not imported",
	};
	const earlier = $derived(imports.filter((i) => i.id !== current?.id).slice(0, 5));
	// Only the pipeline's id, so progress updates don't reopen the stream.
	const following = $derived(current && unfinished(current.pipeline) ? current.pipeline.id : "");

	/** Show an import in full, with first choices for its values once it's checked. */
	function show(found: LabelImport | null) {
		const before = current;
		current = found;
		if (found?.state === "ready" && (before?.id !== found.id || before.state !== "ready")) {
			targets = defaultTargets(values(found), classes);
			overwrite = false;
		}
	}

	function values(found: LabelImport): number[] {
		return (found.values ?? []).map((v) => v.value);
	}

	async function loadClasses() {
		classes = await api<LabelClass[]>(`/api/projects/${pid}/labels/classes`);
	}

	/** Fetch the imports, and the one shown again (the list holds only the newest). */
	async function refresh() {
		const shown = current?.id;
		try {
			imports = await api<LabelImport[]>(base);
			const found =
				imports.find((i) => i.id === shown) ?? (shown ? await api<LabelImport>(`${base}/${shown}`).catch(() => null) : null);
			if (found?.state === "done" && current?.state !== "done") onimported?.();
			show(found ?? imports.find((i) => ACTIVE.includes(i.state)) ?? null);
		} catch (e) {
			error = message(e);
		}
	}

	$effect(() => {
		if (!pid) return;
		untrack(() => {
			// The classes first: a checked file's first choices use them.
			loadClasses()
				.catch((e: unknown) => (error = message(e)))
				.then(() => refresh());
		});
	});

	// Follow the check, then the import, until it ends. The server answers
	// 204 for a pipeline that has ended, which closes the stream; a stream
	// that breaks otherwise (a proxy error while the server restarts, say)
	// opens again shortly.
	$effect(() => {
		const id = following;
		void reconnect;
		if (!id) return;
		let retry: ReturnType<typeof setTimeout> | undefined;
		const source = new EventSource(`/api/projects/${untrack(() => pid)}/pipelines/${id}/events`);
		source.addEventListener("status", (event) => {
			const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
			if (current?.pipeline.id === update.id) current = { ...current, pipeline: update };
			if (!unfinished(update)) void refresh();
		});
		source.addEventListener("error", () => {
			if (source.readyState !== EventSource.CLOSED) return;
			void refresh().then(() => {
				// Once it's over this effect ends, and the extra bump does nothing.
				retry = setTimeout(() => (reconnect += 1), 3000);
			});
		});
		return () => {
			source.close();
			clearTimeout(retry);
		};
	});

	/** Upload a picked or dropped file (resuming an unfinished upload of it), and have it checked. */
	async function choose(chosen: File | undefined) {
		if (!chosen || uploading || busy) return;
		if (!labelFile(chosen.name)) {
			error = `${chosen.name} isn't a TIFF, PNG, or zip file.`;
			return;
		}
		error = "";
		uploading = chosen.name;
		uploaded = 0;
		try {
			const uploads = await api<Upload[]>(`/api/projects/${pid}/uploads`);
			const existing = uploads.find(
				(u) => u.state === "uploading" && u.filename === chosen.name && u.size === chosen.size,
			);
			const done = await uploadFile(pid, chosen, (fraction) => (uploaded = fraction), existing);
			const checking = await api<LabelImport>(base, { body: { upload_id: done.id } }).catch(async (e: unknown) => {
				// Nothing will check the file, so give its storage back. If the
				// check started after all, its job holds the file and this is
				// refused; the list then shows it.
				await api(`/api/projects/${pid}/uploads/${done.id}`, { method: "DELETE" }).catch(() => {});
				await refresh();
				throw e;
			});
			imports = [checking, ...imports.filter((i) => i.id !== checking.id)];
			show(checking);
		} catch (e) {
			error = message(e);
		} finally {
			uploading = null;
		}
	}

	function dropped(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		void choose(event.dataTransfer?.files?.[0]);
	}

	function target(value: number): Target {
		return targets.get(value) ?? { kind: "skip" };
	}

	function setTarget(value: number, next: Target) {
		targets = new Map(targets).set(value, next);
	}

	/** A choice from a value's menu: unlabeled, a label value, or a new class. */
	function pick(value: number, choice: string) {
		if (choice === "skip") return setTarget(value, { kind: "skip" });
		if (choice !== "new") return setTarget(value, { kind: "label", label: Number(choice) });
		const made = [...targets.values()].flatMap((t) => (t.kind === "new" ? [t.color] : []));
		const color = nextColor([...classes.map((c) => c.color), ...made]);
		setTarget(value, { kind: "new", name: String(value), color });
	}

	/** Every value besides 0 as a new class named after it. */
	function eachNew() {
		if (current) targets = newClasses(values(current), classes);
	}

	function choice(chosen: Target): string {
		return chosen.kind === "label" ? String(chosen.label) : chosen.kind;
	}

	async function start() {
		if (!current) return;
		const body = startBody(targets, overwrite);
		if (typeof body === "string") {
			error = body;
			return;
		}
		busy = true;
		error = "";
		try {
			show(await api<LabelImport>(`${base}/${current.id}/start`, { body }));
			await loadClasses();
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}

	/** Stop checking or importing, or drop a checked file; labels already brought in stay. */
	async function discard() {
		if (!current) return;
		const importing = current.state === "importing";
		if (importing && !confirm("Stop importing? Labels already brought in stay.")) return;
		busy = true;
		error = "";
		try {
			await api(`${base}/${current.id}`, { method: "DELETE" });
			if (!importing) current = null;
			await refresh();
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}

	function percent(fraction: number): string {
		return `${Math.round(fraction * 100)}%`;
	}

	function size(shape: [number, number, number]): string {
		const [z, y, x] = shape;
		return `${x} × ${y} × ${z}`;
	}
</script>

<!-- A file dropped just outside the drop zone shouldn't open in the tab. -->
<svelte:window
	ondragover={(e) => {
		if (e.defaultPrevented || !e.dataTransfer) return;
		e.preventDefault();
		e.dataTransfer.dropEffect = "none";
	}}
	ondrop={(e) => e.preventDefault()}
/>

<section class="panel" id="import-labels">
	<h2 class="panel-title">Import labels</h2>
	<div class="flex flex-col gap-3 p-3">
		{#if current}
			<div class="flex items-center gap-2">
				<span class="truncate font-medium" title={current.filename}>{current.filename}</span>
				<span class="ml-auto shrink-0 text-2xs text-ink-dim">{STATES[current.state]}</span>
			</div>
			{#if current.state === "checking" || current.state === "importing"}
				<div class="flex items-center gap-2">
					<progress class="h-1.5 flex-1" max="1" value={current.pipeline.progress}></progress>
					<span class="font-mono text-2xs text-ink-dim">{percent(current.pipeline.progress)}</span>
					<button class="btn btn-ghost" onclick={discard} disabled={busy} aria-label="Stop" title="Stop">
						<X size={13} />
					</button>
				</div>
				<p class="text-2xs text-ink-dim">
					{current.state === "checking"
						? "Checking it's the image's size and finding the values in it."
						: "Bringing the labels in, a chunk at a time. They show in the annotator as they land."}
				</p>
			{:else if current.state === "ready" && current.values && current.shape_zyx}
				{@const found = current.values}
				<p class="text-ink-dim">
					{size(current.shape_zyx)} voxels with {found.length}
					{found.length === 1 ? "value" : "values"}. Choose what each one becomes.
				</p>
				<div class="max-h-72 overflow-y-auto rounded-sm border border-edge">
					<table class="w-full text-2xs">
						<thead class="sticky top-0 z-10 bg-panel-header text-ink-dim">
							<tr>
								<th class="px-2 py-1 text-left font-normal">Value</th>
								<th class="px-2 py-1 text-right font-normal">Voxels</th>
								<th class="px-2 py-1 text-left font-normal">Becomes</th>
							</tr>
						</thead>
						<tbody>
							{#each found as { value, voxels } (value)}
								{@const chosen = target(value)}
								<tr class="border-t border-edge align-top">
									<td class="px-2 py-1.5 font-mono">{value}</td>
									<td class="px-2 py-1.5 text-right font-mono text-ink-dim">{voxels.toLocaleString()}</td>
									<td class="px-2 py-1">
										<div class="flex flex-col gap-1">
											<select
												class="field"
												aria-label="What {value} becomes"
												value={choice(chosen)}
												onchange={(e) => pick(value, e.currentTarget.value)}
											>
												<option value="skip">Unlabeled</option>
												<option value={String(BACKGROUND)}>Background</option>
												{#each classes as label (label.value)}
													<option value={String(label.value)}>{label.name}</option>
												{/each}
												<option value="new">New class</option>
											</select>
											{#if chosen.kind === "new"}
												<div class="flex gap-1">
													<input
														class="h-6 w-7 shrink-0 cursor-pointer rounded-sm border border-edge bg-field p-0.5"
														type="color"
														aria-label="Color of the new class for {value}"
														value={chosen.color}
														oninput={(e) => setTarget(value, { ...chosen, color: e.currentTarget.value })}
													/>
													<input
														class="field"
														aria-label="Name of the new class for {value}"
														maxlength="100"
														value={chosen.name}
														oninput={(e) => setTarget(value, { ...chosen, name: e.currentTarget.value })}
													/>
												</div>
											{/if}
										</div>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<button class="btn btn-ghost self-start" onclick={eachNew}>A new class for each value</button>
				<label class="flex items-start gap-1.5">
					<input class="mt-0.5" type="checkbox" bind:checked={overwrite} />
					<span>
						Replace labels already there
						<span class="block text-2xs text-ink-faint">
							{overwrite ? "Voxels people labeled take the file's labels." : "Off: only voxels nobody labeled are filled."}
						</span>
					</span>
				</label>
				<div class="flex gap-1.5">
					<button class="btn btn-primary" onclick={start} disabled={busy}>Import</button>
					<button class="btn btn-ghost" onclick={discard} disabled={busy}>Discard</button>
				</div>
			{:else}
				{#if current.state === "done"}
					<p class="text-ink-dim">The labels are in. Train on them from the <a href="/p/{pid}/models">Models</a> page.</p>
				{:else if current.state === "failed"}
					<p class="error" role="alert">{current.error ?? "It failed."}</p>
				{:else if current.state === "cancelled"}
					<p class="text-ink-dim">Stopped. Labels already brought in stay.</p>
				{:else}
					<p class="text-ink-dim">The file is gone now; upload it again to import it.</p>
				{/if}
				<button class="btn self-start" onclick={() => (current = null)}>Import another file</button>
			{/if}
		{:else if hasImage === false}
			<p class="text-ink-dim">
				Labels go on the project's image, and it has none yet. Ingest a scan from the <a href="/p/{pid}">Overview</a> page
				first.
			</p>
		{:else if hasImage}
			<p class="text-ink-dim">
				Labels made elsewhere, as a TIFF stack or a zip of TIFF or PNG slices the size of the image. You choose what
				each value means before they come in.
			</p>
			<!-- Its children ignore the pointer, so dragging over them doesn't count as leaving. -->
			<label
				class="flex cursor-pointer flex-col items-center gap-1.5 rounded-sm border border-dashed p-4 text-center transition-colors *:pointer-events-none
					has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-accent
					{dragging ? 'border-accent bg-accent-soft/40' : 'border-line bg-field hover:border-ink-faint'}"
				ondragover={(e) => {
					e.preventDefault();
					dragging = true;
				}}
				ondragleave={() => (dragging = false)}
				ondrop={dropped}
			>
				<FileUp size={18} class="text-ink-faint" />
				<span>{uploading ?? "Drop a label file here, or click to choose"}</span>
				<span class="text-2xs text-ink-faint">.tif, .tiff, .png, or .zip</span>
				<input
					class="sr-only"
					type="file"
					accept=".tif,.tiff,.png,.zip"
					onchange={(e) => choose(e.currentTarget.files?.[0])}
					disabled={!!uploading}
				/>
			</label>
			{#if uploading}
				<div class="flex items-center gap-2">
					<progress class="h-1.5 flex-1" max="1" value={uploaded}></progress>
					<span class="font-mono text-2xs text-ink-dim">{percent(uploaded)}</span>
				</div>
			{/if}
		{/if}
		{#if error}<p class="error" role="alert">{error}</p>{/if}
	</div>
	{#if earlier.length > 0}
		<ul class="divide-y divide-edge border-t border-edge">
			{#each earlier as item (item.id)}
				<li class="flex flex-col gap-0.5 px-3 py-1.5">
					<div class="flex items-center gap-2">
						<span
							class="size-1.5 shrink-0 rounded-full {item.state === 'done'
								? 'bg-ok'
								: item.state === 'failed'
									? 'bg-danger'
									: ACTIVE.includes(item.state)
										? 'bg-warn'
										: 'bg-ink-faint'}"
						></span>
						{#if ACTIVE.includes(item.state)}
							<button class="truncate text-left text-accent-hover hover:underline" onclick={() => show(item)}>{item.filename}</button>
						{:else}
							<span class="truncate">{item.filename}</span>
						{/if}
						<span class="ml-auto shrink-0 text-2xs text-ink-faint">{STATES[item.state]}</span>
					</div>
					{#if item.state === "failed" && item.error}<span class="text-2xs text-danger">{item.error}</span>{/if}
				</li>
			{/each}
		</ul>
	{/if}
</section>
