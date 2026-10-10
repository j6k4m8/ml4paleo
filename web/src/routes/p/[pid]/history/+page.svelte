<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import CheckCheck from "@lucide/svelte/icons/check-check";
	import Eraser from "@lucide/svelte/icons/eraser";
	import FileInput from "@lucide/svelte/icons/file-input";
	import Locate from "@lucide/svelte/icons/locate";
	import Pencil from "@lucide/svelte/icons/pencil";
	import Pentagon from "@lucide/svelte/icons/pentagon";
	import Redo2 from "@lucide/svelte/icons/redo-2";
	import RotateCcw from "@lucide/svelte/icons/rotate-ccw";
	import RotateCcwClock from "@lucide/svelte/icons/rotate-ccw-clock";
	import Undo2 from "@lucide/svelte/icons/undo-2";
	import X from "@lucide/svelte/icons/x";
	import { untrack } from "svelte";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import {
		type Action,
		action,
		ago,
		type Entry,
		everything,
		link,
		noteAuthors,
		origin,
		people,
		place,
		refreshed,
		SOURCES,
		type Toggle,
		toggles,
		what,
		who,
	} from "#lib/history.ts";
	import { whileVisible } from "#lib/refresh.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import ProjectTabs from "#lib/ui/ProjectTabs.svelte";

	// Entries asked for at a time, and the server's most, which reloads ask for.
	const PAGE = 100;
	const MOST = 500;

	const ICONS: Record<Action, typeof Brush> = {
		brush: Brush,
		eraser: Eraser,
		polygon: Pentagon,
		"polygon-erase": Pentagon,
		accept: CheckCheck,
		decline: X,
		"restore-declined": RotateCcw,
		import: FileInput,
		edit: Pencil,
		undo: Undo2,
		redo: Redo2,
	};

	/** Whose and which entries are listed; empty for everyone and every source. */
	interface Filters {
		pid: string;
		person: string;
		source: string;
	}

	const pid = $derived(page.params.pid ?? "");
	let projectName = $state("");
	let members: { user_id: string; username: string }[] = $state([]);
	// Everyone listed so far, by id, so people who left can be picked too.
	let authors: Record<string, string> = $state({});
	let person = $state("");
	let source = $state("");
	let entries: Entry[] = $state([]);
	// Whether older entries may be left to load.
	let more = $state(false);
	let loaded = $state(false);
	let loadingOlder = $state(false);
	// The edit an undo or redo is on its way for.
	let sending: number | null = $state(null);
	let loadError = $state("");
	let error = $state("");
	let now = $state(Date.now());
	// The list shown: its filters, a signal that aborts once it starts over,
	// and how to reload it.
	let listed: { filters: Filters; signal: AbortSignal } | null = null;
	let reload: () => Promise<void> = async () => {};
	// Reloads started, so only the latest one's answer is shown.
	let reloads = 0;

	const buttons = $derived(toggles(entries));
	const everyone = $derived(people(members, authors));

	$effect(() => {
		crumbs.set([{ label: "Projects", href: "/projects" }, { label: projectName || "…", href: `/p/${pid}` }, { label: "History" }]);
	});

	$effect(() => {
		api<{ name: string; members: { user_id: string; username: string }[] | null }>(`/api/projects/${pid}`).then(
			(project) => {
				projectName = project.name;
				members = project.members ?? [];
			},
			() => {},
		);
	});

	/** Entries between ops `after` and `before`, newest first, `limit` of them at most. */
	function fetchEntries(
		filters: Filters,
		range: { limit: number; before?: number | null; after?: number },
		signal: AbortSignal,
	): Promise<Entry[]> {
		const query = new URLSearchParams({ limit: String(range.limit) });
		if (range.before) query.set("before", String(range.before));
		if (range.after !== undefined) query.set("after", String(range.after));
		if (filters.person) query.set("user_id", filters.person);
		if (filters.source) query.set("source", filters.source);
		return api<Entry[]>(`/api/projects/${filters.pid}/labels/ops?${query}`, { signal });
	}

	// The newest entries, and every one shown kept up to date while the page
	// is visible; the list starts over for another project or other filters.
	$effect(() => {
		const filters: Filters = { pid, person, source };
		const controller = new AbortController();
		const { signal } = controller;
		if (listed?.filters.pid !== filters.pid) authors = {};
		listed = { filters, signal };
		entries = [];
		more = false;
		loaded = false;
		loadError = "";
		error = "";
		reload = async () => {
			const reloading = ++reloads;
			const oldest = entries.at(-1)?.seq;
			try {
				const fresh =
					oldest === undefined
						? await fetchEntries(filters, { limit: PAGE }, signal)
						: await everything((before) => fetchEntries(filters, { limit: MOST, before, after: oldest - 1 }, signal), MOST);
				if (reloading !== reloads) return;
				({ entries, more } = refreshed(fresh, oldest, { entries, more }, PAGE));
				noteAuthors(authors, fresh);
				now = Date.now();
				loadError = "";
			} catch (e) {
				if (signal.aborted || reloading !== reloads) return;
				loadError = message(e);
			}
			loaded = true;
		};
		untrack(() => void reload());
		const stop = whileVisible(() => void reload());
		return () => {
			controller.abort();
			stop();
		};
	});

	async function loadOlder() {
		const list = listed;
		const last = entries.at(-1);
		if (!list || !last || loadingOlder) return;
		loadingOlder = true;
		error = "";
		try {
			const older = await fetchEntries(list.filters, { limit: PAGE, before: last.seq }, list.signal);
			// Unless the list started over, or a reload changed it, meanwhile.
			if (list === listed && entries.at(-1)?.seq === last.seq) {
				entries = [...entries, ...older];
				more = older.length === PAGE;
				noteAuthors(authors, older);
			}
		} catch (e) {
			if (!list.signal.aborted) error = message(e);
		} finally {
			loadingOlder = false;
		}
	}

	async function toggle({ seq, action }: Toggle) {
		if (sending !== null) return;
		sending = seq;
		error = "";
		try {
			await api(`/api/projects/${pid}/labels/ops/${seq}/${action}`, { body: { client_op_id: crypto.randomUUID() } });
		} catch (e) {
			error = e instanceof ApiError && e.status === 409 ? `#${seq} is already ${action === "undo" ? "undone" : "live"}.` : message(e);
		}
		await reload();
		sending = null;
	}
</script>

<ProjectTabs {pid} />

<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
	<div class="flex flex-wrap items-center gap-3">
		<h1>History</h1>
		<div class="ml-auto flex gap-2">
			<select class="field w-36" bind:value={person} aria-label="Person">
				<option value="">Everyone</option>
				{#each everyone as someone (someone.id)}<option value={someone.id}>{someone.name}</option>{/each}
			</select>
			<select class="field w-44" bind:value={source} aria-label="Source">
				<option value="">Every source</option>
				{#each SOURCES as [value, label] (value)}<option value={String(value)}>{label}</option>{/each}
			</select>
		</div>
	</div>
	{#if loadError}<p class="error" role="alert">{loadError}</p>{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}

	{#if entries.length > 0}
		<section class="panel overflow-x-auto">
			<table class="w-full min-w-max text-left">
				<thead class="text-2xs text-ink-dim">
					<tr class="border-b border-edge">
						<th class="py-1.5 pr-3 pl-3 font-normal">#</th>
						<th class="py-1.5 pr-3 font-normal">When</th>
						<th class="py-1.5 pr-3 font-normal">Who</th>
						<th class="py-1.5 pr-3 font-normal">What</th>
						<th class="py-1.5 pr-3 font-normal">Source</th>
						<th class="py-1.5 pr-3 font-normal">Where</th>
						<th class="py-1.5 pr-3 font-normal">State</th>
						<th></th>
					</tr>
				</thead>
				<tbody class="divide-y divide-edge">
					{#each entries as entry (entry.seq)}
						{@const Icon = ICONS[action(entry)]}
						{@const at = place(entry.bbox)}
						{@const button = buttons.get(entry.seq)}
						<tr class="align-top {entry.kind === 'edit' && !entry.live ? 'text-ink-faint' : ''}">
							<td class="py-1.5 pr-3 pl-3 font-mono text-2xs text-ink-faint">{entry.seq}</td>
							<td class="py-1.5 pr-3 whitespace-nowrap text-ink-dim" title={new Date(entry.created_at).toLocaleString()}>
								{ago(entry.created_at, now)}
							</td>
							<td class="py-1.5 pr-3">{who(entry)}</td>
							<td class="py-1.5 pr-3">
								<span class="flex items-center gap-1.5 whitespace-nowrap"><Icon size={13} class="shrink-0 text-ink-dim" />{what(entry)}</span>
								{#if entry.target}
									<span class="block pl-5 text-2xs text-ink-dim">{what(entry.target)} by {who(entry.target)}</span>
								{/if}
							</td>
							<td class="py-1.5 pr-3">{origin(entry)}</td>
							<td class="py-1.5 pr-3 font-mono text-2xs whitespace-nowrap">
								{at.size} <span class="font-sans text-ink-faint">at</span>
								<span class="text-axis-x">x</span>{at.x}
								<span class="text-axis-y">y</span>{at.y}
								<span class="text-axis-z">z</span>{at.z}
							</td>
							<td class="py-1.5 pr-3 whitespace-nowrap">
								{#if entry.kind === "edit"}
									<span class="flex items-center gap-1.5">
										<span class="size-1.5 rounded-full {entry.live ? 'bg-ok' : 'bg-ink-faint'}"></span>
										{entry.live ? "Live" : "Undone"}
									</span>
								{/if}
							</td>
							<td class="py-1 pr-3">
								<div class="flex justify-end gap-1">
									{#if button}
										{@const label = `${button.action === "undo" ? "Undo" : "Redo"} #${button.seq}`}
										<button class="btn" disabled={sending !== null} title={label} aria-label={label} onclick={() => toggle(button)}>
											{#if button.action === "undo"}<Undo2 size={12} /> Undo{:else}<Redo2 size={12} /> Redo{/if}
										</button>
									{/if}
									<a
										class="btn btn-ghost hover:no-underline"
										href={link(pid, entry)}
										title="Open the annotator here"
										aria-label="Open #{entry.seq} in the annotator"
									>
										<Locate size={12} /> Open
									</a>
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<div class="border-t border-edge p-2.5">
				{#if more}
					<button class="btn" disabled={loadingOlder} onclick={loadOlder}>{loadingOlder ? "Loading…" : "Load older"}</button>
				{:else}
					<span class="text-2xs text-ink-faint">No older edits.</span>
				{/if}
			</div>
		</section>
	{:else if loaded && !loadError}
		<div class="panel grid place-items-center gap-2 p-10 text-center">
			<RotateCcwClock size={28} class="text-ink-faint" />
			<p class="muted">{person || source ? "No edits match." : "No label edits yet."}</p>
		</div>
	{/if}
</div>
