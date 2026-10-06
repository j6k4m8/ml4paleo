<script lang="ts">
	import Copy from "@lucide/svelte/icons/copy";
	import RefreshCw from "@lucide/svelte/icons/refresh-cw";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import { parseJobId } from "#lib/v1.ts";

	type Override = Partial<Record<"storage_gb" | "trained_models", number | null>>;

	interface Account {
		id: string;
		username: string;
		email: string | null;
		is_admin: boolean;
		status: "active" | "unverified" | "disabled";
		created_at: string;
		last_login_at: string | null;
		storage_bytes_used: number;
		trained_models_used: number;
		quota_override: Override;
	}

	interface QuotaRequest {
		id: string;
		username: string;
		message: string;
		created_at: string;
		quota_override: Override;
	}

	interface WorkerRow {
		id: string;
		name: string;
		pool: string;
		status: string;
		online: boolean;
		running_jobs: number;
		last_seen_at: string | null;
	}

	interface JobRow {
		id: string;
		kind: string;
		status: string;
		progress: number;
		attempts: number;
		max_attempts: number;
		error: string | null;
		worker: string | null;
		created_at: string;
	}

	const ACTIVE = ["blocked", "queued", "leased"];

	let signupMode = $state<"open" | "invite">("open");
	let inviteEmail = $state("");
	let inviteUrl = $state("");
	let requests: QuotaRequest[] = $state([]);
	let grants: Record<string, { storage: string; models: string }> = $state({});
	let accounts: Account[] = $state([]);
	let search = $state("");
	let editing: string | null = $state(null);
	let limits = $state({ storage: "", models: "" });
	let workers: WorkerRow[] = $state([]);
	let workerName = $state("");
	let workerToken = $state("");
	let jobs: JobRow[] = $state([]);
	let jobStatus = $state("");
	let releaseId = $state("");
	let notice = $state("");
	let error = $state("");

	$effect(() => crumbs.set([{ label: "Administration" }]));

	$effect(() => {
		if (session.current?.user.is_admin) load();
	});

	async function run(action: () => Promise<unknown>, done?: string) {
		error = "";
		notice = "";
		try {
			await action();
			if (done) notice = done;
		} catch (e) {
			error = message(e);
		}
	}

	async function load() {
		await run(async () => {
			signupMode = (await api<{ signup_mode: "open" | "invite" }>("/api/admin/settings")).signup_mode;
			[requests, workers] = await Promise.all([
				api<QuotaRequest[]>("/api/admin/quota-requests"),
				api<WorkerRow[]>("/api/admin/workers"),
			]);
			await Promise.all([loadAccounts(), loadJobs()]);
		});
	}

	async function loadAccounts() {
		accounts = await api<Account[]>(`/api/admin/users?q=${encodeURIComponent(search)}`);
	}

	async function loadJobs() {
		jobs = await api<JobRow[]>(`/api/admin/jobs?limit=50${jobStatus ? `&status=${jobStatus}` : ""}`);
	}

	/** A limit as typed: empty keeps the default, "unlimited" has none. */
	function parseLimit(text: string): number | null | undefined {
		const value = text.trim().toLowerCase();
		if (!value) return undefined;
		if (value === "unlimited" || value === "none") return null;
		const number = Number(value);
		if (!Number.isFinite(number) || number < 0) throw new Error(`"${text}" isn't a limit.`);
		return number;
	}

	function override(storage: string, models: string): Override {
		const result: Override = {};
		const gb = parseLimit(storage);
		const count = parseLimit(models);
		if (gb !== undefined) result.storage_gb = gb;
		if (count !== undefined) result.trained_models = count === null ? null : Math.round(count);
		return result;
	}

	const showLimit = (value: number | null | undefined, unit: string) =>
		value === undefined ? "default" : value === null ? "unlimited" : `${value}${unit}`;
	const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes < 1024 ** 3 ? 2 : 1)} GB`;
	const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "never");

	function copy(text: string) {
		navigator.clipboard?.writeText(text).then(
			() => (notice = "Copied."),
			() => {},
		);
	}
</script>

{#if !session.current?.user.is_admin}
	<p class="p-6 text-ink-dim">Administration is for admins.</p>
{:else}
	<div class="mx-auto flex max-w-6xl flex-col gap-4 p-6">
		<div class="flex items-center gap-2">
			<h1>Administration</h1>
			<button class="btn btn-ghost ml-auto" onclick={load}><RefreshCw size={13} /> Refresh</button>
		</div>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		{#if notice}<p class="text-ok" role="status">{notice}</p>{/if}

		<div class="grid gap-4 lg:grid-cols-2">
			<section class="panel self-start">
				<h2 class="panel-title">Sign-up</h2>
				<div class="flex flex-col gap-3 p-3">
					<label class="label">
						Who can make an account
						<select
							class="field"
							bind:value={signupMode}
							onchange={() =>
								run(() => api("/api/admin/settings", { method: "PUT", body: { signup_mode: signupMode } }), "Saved.")}
						>
							<option value="open">Anyone</option>
							<option value="invite">People with an invite link</option>
						</select>
					</label>
					<form
						class="flex items-end gap-2"
						onsubmit={(event) => {
							event.preventDefault();
							run(async () => {
								const invite = await api<{ url: string }>("/api/admin/invites", {
									body: inviteEmail ? { email: inviteEmail } : {},
								});
								inviteUrl = invite.url;
								inviteEmail = "";
							});
						}}
					>
						<label class="label flex-1">
							Invite (an email address is optional; with mail set up, it gets the link)
							<input class="field" type="email" bind:value={inviteEmail} placeholder="someone@example.org" />
						</label>
						<button class="btn">Make an invite link</button>
					</form>
					{#if inviteUrl}
						<div class="flex items-center gap-2 rounded-sm border border-edge bg-field p-2">
							<code class="min-w-0 flex-1 truncate text-2xs">{inviteUrl}</code>
							<button class="btn btn-ghost" onclick={() => copy(inviteUrl)} aria-label="Copy the invite link"><Copy size={13} /></button>
						</div>
						<p class="text-2xs text-ink-dim">It works once, for two weeks.</p>
					{/if}
				</div>
			</section>

			<section class="panel self-start">
				<h2 class="panel-title">Requests for more</h2>
				<div class="flex flex-col gap-2 p-3">
					{#each requests as request (request.id)}
						{@const grant = (grants[request.id] ??= { storage: "", models: "" })}
						<div class="flex flex-col gap-2 rounded-sm border border-edge bg-field p-2">
							<div class="flex items-center gap-2">
								<span class="font-medium">{request.username}</span>
								<span class="text-2xs text-ink-dim">{when(request.created_at)}</span>
							</div>
							<p class="whitespace-pre-wrap">{request.message}</p>
							<div class="flex flex-wrap items-end gap-2">
								<label class="label">
									Storage (GB)
									<input class="field w-28" bind:value={grant.storage} placeholder={showLimit(request.quota_override.storage_gb, "")} />
								</label>
								<label class="label">
									Models
									<input class="field w-24" bind:value={grant.models} placeholder={showLimit(request.quota_override.trained_models, "")} />
								</label>
								<button
									class="btn btn-primary"
									onclick={() =>
										run(async () => {
											await api(`/api/admin/quota-requests/${request.id}`, {
												body: { decision: "grant", quota_override: override(grant.storage, grant.models) },
											});
											await load();
										}, `Granted ${request.username} more.`)}
								>
									Grant
								</button>
								<button
									class="btn btn-ghost"
									onclick={() =>
										run(async () => {
											await api(`/api/admin/quota-requests/${request.id}`, { body: { decision: "decline" } });
											await load();
										})}
								>
									Decline
								</button>
							</div>
						</div>
					{:else}
						<p class="text-ink-dim">No open requests.</p>
					{/each}
				</div>
			</section>
		</div>

		<section class="panel">
			<h2 class="panel-title">Accounts</h2>
			<div class="flex flex-col gap-2 p-3">
				<form
					class="flex items-end gap-2"
					onsubmit={(event) => {
						event.preventDefault();
						run(loadAccounts);
					}}
				>
					<label class="label flex-1">
						Find by the start of a username or email
						<input class="field" bind:value={search} />
					</label>
					<button class="btn">Find</button>
				</form>
				<div class="overflow-x-auto">
					<table class="w-full text-left">
						<thead class="text-2xs text-ink-dim">
							<tr>
								<th class="py-1 pr-3 font-normal">Account</th>
								<th class="py-1 pr-3 font-normal">Status</th>
								<th class="py-1 pr-3 font-normal">Storage</th>
								<th class="py-1 pr-3 font-normal">Models</th>
								<th class="py-1 pr-3 font-normal">Last sign-in</th>
								<th></th>
							</tr>
						</thead>
						<tbody class="divide-y divide-edge">
							{#each accounts as account (account.id)}
								<tr class="align-top">
									<td class="py-1.5 pr-3">
										<div class="font-medium">
											{account.username}
											{#if account.is_admin}<span class="text-2xs text-accent-hover">admin</span>{/if}
										</div>
										<div class="text-2xs text-ink-dim">{account.email ?? "no email"}</div>
									</td>
									<td class="py-1.5 pr-3" class:text-danger={account.status === "disabled"}>{account.status}</td>
									<td class="py-1.5 pr-3 font-mono text-2xs">
										{gb(account.storage_bytes_used)} / {showLimit(account.quota_override.storage_gb, " GB")}
									</td>
									<td class="py-1.5 pr-3 font-mono text-2xs">
										{account.trained_models_used} / {showLimit(account.quota_override.trained_models, "")}
									</td>
									<td class="py-1.5 pr-3 text-2xs text-ink-dim">{when(account.last_login_at)}</td>
									<td class="py-1.5">
										<div class="flex justify-end gap-1">
											<button
												class="btn btn-ghost"
												onclick={() => {
													editing = editing === account.id ? null : account.id;
													limits = {
														storage: account.quota_override.storage_gb === null ? "unlimited" : String(account.quota_override.storage_gb ?? ""),
														models: account.quota_override.trained_models === null ? "unlimited" : String(account.quota_override.trained_models ?? ""),
													};
												}}
											>
												Limits
											</button>
											{#if account.status === "disabled"}
												<button
													class="btn btn-ghost"
													onclick={() =>
														run(async () => {
															await api(`/api/admin/users/${account.id}/status`, { method: "PUT", body: { status: "active" } });
															await loadAccounts();
														})}
												>
													Enable
												</button>
											{:else if account.id !== session.current?.user.id}
												<button
													class="btn btn-danger"
													onclick={() =>
														run(async () => {
															await api(`/api/admin/users/${account.id}/status`, { method: "PUT", body: { status: "disabled" } });
															await loadAccounts();
														}, `${account.username} can't sign in now.`)}
												>
													Disable
												</button>
											{/if}
										</div>
									</td>
								</tr>
								{#if editing === account.id}
									<tr>
										<td colspan="6" class="pb-2">
											<form
												class="flex flex-wrap items-end gap-2 rounded-sm border border-edge bg-field p-2"
												onsubmit={(event) => {
													event.preventDefault();
													run(async () => {
														await api(`/api/admin/users/${account.id}/quota`, {
															method: "PUT",
															body: override(limits.storage, limits.models),
														});
														editing = null;
														await loadAccounts();
													}, "Saved.");
												}}
											>
												<label class="label">
													Storage (GB)
													<input class="field w-28" bind:value={limits.storage} placeholder="default" />
												</label>
												<label class="label">
													Models
													<input class="field w-24" bind:value={limits.models} placeholder="default" />
												</label>
												<button class="btn btn-primary">Save</button>
												<p class="text-2xs text-ink-dim">Leave a limit empty for the default, or type "unlimited".</p>
											</form>
										</td>
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		</section>

		<div class="grid gap-4 lg:grid-cols-2">
			<section class="panel self-start">
				<h2 class="panel-title">Workers</h2>
				<div class="flex flex-col gap-2 p-3">
					<ul class="flex flex-col divide-y divide-edge">
						{#each workers as worker (worker.id)}
							<li class="flex items-center gap-2 py-1.5">
								<span class="size-2 shrink-0 rounded-full" class:bg-ok={worker.online} class:bg-ink-faint={!worker.online}></span>
								<span class="font-medium" class:text-ink-faint={worker.status === "revoked"}>{worker.name}</span>
								<span class="text-2xs text-ink-dim">
									{#if worker.status === "revoked"}
										revoked
									{:else}
										{worker.pool} · {worker.running_jobs} running · seen {when(worker.last_seen_at)}
									{/if}
								</span>
								{#if worker.pool !== "local" && worker.status !== "revoked"}
									<button
										class="btn btn-ghost ml-auto"
										onclick={() =>
											run(async () => {
												await api(`/api/admin/workers/${worker.id}`, { method: "DELETE" });
												workers = await api<WorkerRow[]>("/api/admin/workers");
											}, `${worker.name} can't take jobs now.`)}
									>
										Revoke
									</button>
								{/if}
							</li>
						{:else}
							<li class="text-ink-dim">No workers yet.</li>
						{/each}
					</ul>
					<form
						class="flex items-end gap-2"
						onsubmit={(event) => {
							event.preventDefault();
							run(async () => {
								const made = await api<{ token: string }>("/api/admin/workers", { body: { name: workerName } });
								workerToken = made.token;
								workerName = "";
								workers = await api<WorkerRow[]>("/api/admin/workers");
							});
						}}
					>
						<label class="label flex-1">
							Add a worker on another machine (letters, digits, - _ .)
							<input class="field" bind:value={workerName} pattern="[A-Za-z0-9_.\-]+" required />
						</label>
						<button class="btn">Make its token</button>
					</form>
					{#if workerToken}
						<div class="flex flex-col gap-1 rounded-sm border border-edge bg-field p-2">
							<p class="text-2xs text-warn">Copy this token now; it isn't shown again.</p>
							<div class="flex items-center gap-2">
								<code class="min-w-0 flex-1 truncate text-2xs">{workerToken}</code>
								<button class="btn btn-ghost" onclick={() => copy(workerToken)} aria-label="Copy the worker token"><Copy size={13} /></button>
							</div>
							<p class="text-2xs text-ink-dim">
								Save it in a file on that machine and start the worker with <code>--server {location.origin}
									--token-file &lt;file&gt;</code>.
							</p>
						</div>
					{/if}
				</div>
			</section>

			<section class="panel self-start">
				<h2 class="panel-title">v1 jobs</h2>
				<form
					class="flex items-end gap-2 p-3"
					onsubmit={(event) => {
						event.preventDefault();
						const id = parseJobId(releaseId);
						if (!id) {
							error = "That isn't a v1 job's id or link.";
							return;
						}
						run(async () => {
							await api(`/api/v1-jobs/${id}/release`, { method: "POST" });
							releaseId = "";
						}, `Released ${id}: its project is deleted, and the job can be claimed again.`);
					}}
				>
					<label class="label flex-1">
						Release a job someone else claimed (deletes their project from it)
						<input class="field font-mono" bind:value={releaseId} placeholder="AB12CD" />
					</label>
					<button class="btn btn-danger">Release</button>
				</form>
			</section>
		</div>

		<section class="panel">
			<div class="panel-title flex items-center gap-2">
				<h2>Jobs</h2>
				<select class="field ml-auto w-36 normal-case" bind:value={jobStatus} onchange={() => run(loadJobs)} aria-label="Show jobs that are">
					<option value="">all</option>
					{#each ["blocked", "queued", "leased", "succeeded", "failed", "cancelled"] as status (status)}
						<option value={status}>{status}</option>
					{/each}
				</select>
			</div>
			<div class="overflow-x-auto p-3">
				<table class="w-full text-left">
					<thead class="text-2xs text-ink-dim">
						<tr>
							<th class="py-1 pr-3 font-normal">Kind</th>
							<th class="py-1 pr-3 font-normal">Status</th>
							<th class="py-1 pr-3 font-normal">Worker</th>
							<th class="py-1 pr-3 font-normal">Made</th>
							<th class="py-1 pr-3 font-normal">Error</th>
							<th></th>
						</tr>
					</thead>
					<tbody class="divide-y divide-edge">
						{#each jobs as job (job.id)}
							<tr class="align-top">
								<td class="py-1.5 pr-3 font-mono text-2xs">{job.kind}</td>
								<td class="py-1.5 pr-3 text-2xs">
									{job.status}{#if job.status === "leased"}&nbsp;{Math.round(job.progress * 100)}%{/if}
									{#if job.attempts > 1}<span class="text-ink-dim"> · try {job.attempts}/{job.max_attempts}</span>{/if}
								</td>
								<td class="py-1.5 pr-3 text-2xs">{job.worker ?? ""}</td>
								<td class="py-1.5 pr-3 text-2xs text-ink-dim">{when(job.created_at)}</td>
								<td class="max-w-80 truncate py-1.5 pr-3 text-2xs text-danger" title={job.error ?? ""}>{job.error ?? ""}</td>
								<td class="py-1.5 text-right">
									{#if ACTIVE.includes(job.status)}
										<button
											class="btn btn-ghost"
											onclick={() =>
												run(async () => {
													await api(`/api/admin/jobs/${job.id}/cancel`, { method: "POST" });
													await loadJobs();
												})}
										>
											Cancel
										</button>
									{/if}
								</td>
							</tr>
						{:else}
							<tr><td colspan="6" class="py-2 text-ink-dim">No jobs.</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>
	</div>
{/if}
