<script lang="ts">
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { Session } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Panel from "#lib/ui/Panel.svelte";

	interface Quota {
		storage_bytes_limit: number | null;
		storage_bytes_used: number;
		trained_models_limit: number | null;
		trained_models_used: number;
		open_request: boolean;
	}

	let current = $state("");
	let next = $state("");
	let passwordError = $state("");
	let passwordDone = $state(false);

	let setup: { secret: string; otpauth_uri: string } | null = $state(null);
	let code = $state("");
	let totpError = $state("");

	let emailOn = $state(false);
	let resent = $state(false);
	let resendError = $state("");
	let address = $state("");
	let addressPassword = $state("");
	let addressError = $state("");
	let addressDone = $state("");

	let quota: Quota | null = $state(null);
	let ask = $state("");
	let askError = $state("");

	const steps = $derived(session.current?.required_steps ?? []);
	const user = $derived(session.current?.user);
	$effect(() => crumbs.set([{ label: "Account" }]));

	$effect(() => {
		api<{ email_enabled: boolean }>("/api/auth/config").then(
			(config) => (emailOn = config.email_enabled),
			() => {},
		);
	});

	// Limits are for accounts that can use the site, so not mid-setup.
	$effect(() => {
		if (session.current && steps.length === 0) loadQuota();
	});

	async function loadQuota() {
		quota = await api<Quota>("/api/me/quota").catch(() => null);
	}

	async function resend() {
		resendError = "";
		try {
			await api("/api/auth/verify-email/resend", { method: "POST" });
			resent = true;
		} catch (e) {
			resendError = message(e);
		}
	}

	async function changeAddress(event: SubmitEvent) {
		event.preventDefault();
		addressError = "";
		addressDone = "";
		try {
			session.current = await api<Session>("/api/auth/email", {
				method: "PUT",
				body: { email: address, current_password: addressPassword },
			});
			addressDone = emailOn ? `Sent a link to ${session.current.user.email} to confirm it.` : "Saved.";
			address = addressPassword = "";
			resent = false;
			await loadQuota();
		} catch (e) {
			addressError = message(e);
		}
	}

	async function askForMore(event: SubmitEvent) {
		event.preventDefault();
		askError = "";
		try {
			await api("/api/me/quota-requests", { body: { message: ask } });
			ask = "";
			await loadQuota();
		} catch (e) {
			askError = message(e);
		}
	}

	async function changePassword(event: SubmitEvent) {
		event.preventDefault();
		passwordError = "";
		try {
			await api("/api/auth/password", { body: { current_password: current, new_password: next } });
			passwordDone = true;
			current = next = "";
			await session.load();
		} catch (e) {
			passwordError = message(e);
		}
	}

	async function startTotp() {
		totpError = "";
		try {
			setup = await api("/api/auth/totp/setup", { method: "POST" });
		} catch (e) {
			totpError = message(e);
		}
	}

	async function confirmTotp(event: SubmitEvent) {
		event.preventDefault();
		totpError = "";
		try {
			await api("/api/auth/totp/confirm", { body: { code } });
			setup = null;
			await session.load();
		} catch (e) {
			totpError = message(e);
		}
	}

	const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes < 1024 ** 3 ? 2 : 1)} GB`;
	const limitOf = (value: number | null, show: (n: number) => string) => (value === null ? "unlimited" : show(value));
</script>

<div class="mx-auto flex max-w-xl flex-col gap-3 p-6">
	<h1>Account</h1>
	{#if steps.length > 0}
		<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">Finish these steps to use ml4paleo.</p>
	{/if}

	<div class="panel overflow-hidden">
		<Panel title="Email">
			{#if user?.email}
				<p>
					{user.email}
					{#if user.email_verified}
						<span class="text-2xs text-ok">confirmed</span>
					{:else}
						<span class="text-2xs text-warn">not confirmed</span>
					{/if}
				</p>
				{#if !user.email_verified}
					{#if emailOn}
						<p class="muted">Open the link we sent to confirm it's yours.</p>
						{#if resent}
							<p class="muted">Sent another one.</p>
						{:else}
							<button class="btn self-start" onclick={resend}>Send the link again</button>
						{/if}
						{#if resendError}<p class="error" role="alert">{resendError}</p>{/if}
					{:else}
						<p class="muted">This site doesn't send email, so an admin confirms addresses.</p>
					{/if}
				{/if}
			{:else}
				<p class="muted">No email address.</p>
			{/if}
			{#if steps.length === 0}
				<form class="flex max-w-sm flex-col gap-3" onsubmit={changeAddress}>
					<label class="label">
						{user?.email ? "New address" : "Address"}
						<input class="field" type="email" bind:value={address} autocomplete="email" required />
					</label>
					<label class="label">
						Current password
						<input class="field" type="password" bind:value={addressPassword} autocomplete="current-password" required />
					</label>
					{#if addressError}<p class="error" role="alert">{addressError}</p>{/if}
					{#if addressDone}<p class="text-ok">{addressDone}</p>{/if}
					<button class="btn self-start">{user?.email ? "Change address" : "Add address"}</button>
				</form>
			{/if}
		</Panel>

		{#if quota}
			<Panel title="Limits">
				{#if session.current?.starter_limits}
					<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">
						These are starter limits. You get the usual ones once your email address is confirmed.
					</p>
				{/if}
				<dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
					<dt class="text-ink-dim">Storage</dt>
					<dd class="font-mono">{gb(quota.storage_bytes_used)} of {limitOf(quota.storage_bytes_limit, gb)}</dd>
					<dt class="text-ink-dim">Trained models</dt>
					<dd class="font-mono">{quota.trained_models_used} of {limitOf(quota.trained_models_limit, String)}</dd>
				</dl>
				{#if quota.open_request}
					<p class="muted">Your request for more is with the admins.</p>
				{:else}
					<form class="flex flex-col gap-3" onsubmit={askForMore}>
						<label class="label">
							Ask for more
							<textarea
								class="field h-16 py-1"
								bind:value={ask}
								maxlength="2000"
								required
								placeholder="What you're working on, and roughly how much you need."
							></textarea>
						</label>
						{#if askError}<p class="error" role="alert">{askError}</p>{/if}
						<button class="btn self-start">Send to the admins</button>
					</form>
				{/if}
			</Panel>
		{/if}

		<Panel title="Password">
			<form class="flex max-w-sm flex-col gap-3" onsubmit={changePassword}>
				<label class="label">
					Current password
					<input class="field" type="password" bind:value={current} autocomplete="current-password" required />
				</label>
				<label class="label">
					New password
					<input class="field" type="password" bind:value={next} autocomplete="new-password" minlength="12" required />
				</label>
				{#if passwordError}<p class="error" role="alert">{passwordError}</p>{/if}
				{#if passwordDone}<p class="text-ok">Password changed.</p>{/if}
				<button class="btn btn-primary self-start">Change password</button>
			</form>
		</Panel>

		{#if steps.includes("set_up_two_factor")}
			<Panel title="Two-factor sign-in">
				{#if setup}
					<p>Add this key to your authenticator app, then enter the code it shows.</p>
					<code class="kbd self-start !text-xs">{setup.secret}</code>
					<a href={setup.otpauth_uri}>Open in an authenticator app</a>
					<form class="flex max-w-sm flex-col gap-3" onsubmit={confirmTotp}>
						<label class="label">Code <input class="field" bind:value={code} inputmode="numeric" autocomplete="one-time-code" required /></label>
						<button class="btn btn-primary self-start">Turn on</button>
					</form>
				{:else}
					<button class="btn btn-primary self-start" onclick={startTotp}>Set up two-factor sign-in</button>
				{/if}
				{#if totpError}<p class="error" role="alert">{totpError}</p>{/if}
			</Panel>
		{/if}
	</div>
</div>
