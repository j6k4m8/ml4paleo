<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { authenticatorQr } from "#lib/authenticator-qr.ts";
	import { safeNext } from "#lib/navigation.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { Session } from "#lib/types.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Panel from "#lib/ui/Panel.svelte";
	import AuthenticatorSetup from "#lib/ui/AuthenticatorSetup.svelte";

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
	let totpBusy = $state(false);
	let qr = $state("");
	let qrError = $state(false);

	// Whether this site sends email (null until known).
	let emailOn: boolean | null = $state(null);
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
	const needsPassword = $derived(steps.includes("change_password"));
	const needsTotp = $derived(steps.includes("set_up_two_factor"));
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
			await returnAfterSetup();
		} catch (e) {
			passwordError = message(e);
		}
	}

	async function startTotp() {
		if (totpBusy) return;
		totpBusy = true;
		totpError = "";
		qrError = false;
		qr = "";
		code = "";
		try {
			setup = await api("/api/auth/totp/setup", { method: "POST" });
			try {
				qr = await authenticatorQr(setup!.otpauth_uri);
			} catch {
				// Keep the same pending key usable through the manual / app-link fallback.
				qrError = true;
			}
		} catch (e) {
			totpError = message(e);
		} finally {
			totpBusy = false;
		}
	}

	async function confirmTotp(event: SubmitEvent) {
		event.preventDefault();
		if (totpBusy) return;
		totpBusy = true;
		totpError = "";
		try {
			await api("/api/auth/totp/confirm", { body: { code } });
			setup = null;
			qr = code = "";
			await session.load();
			await returnAfterSetup();
		} catch (e) {
			totpError = message(e);
		} finally {
			totpBusy = false;
		}
	}

	async function returnAfterSetup() {
		const target = page.url.searchParams.get("next");
		if (target && session.current && session.current.required_steps.length === 0) {
			await goto(safeNext(target, page.url.origin), { replace: true });
		}
	}

	const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes < 1024 ** 3 ? 2 : 1)} GB`;
	const limitOf = (value: number | null, show: (n: number) => string) => (value === null ? "unlimited" : show(value));
</script>

{#snippet passwordForm()}
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
{/snippet}

<div class="mx-auto flex max-w-xl flex-col gap-3 p-6">
	<h1>Account</h1>
	{#if needsTotp || needsPassword}
		<div class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn" role="status">
			{#if needsTotp}<p>Administration requires two-factor sign-in. Set it up below to continue.</p>{/if}
			{#if needsPassword}<p>Change your temporary password before continuing.</p>{/if}
		</div>
	{/if}

	<div class="panel overflow-hidden">
		{#if needsPassword}
			<Panel title="Change your temporary password">
				{@render passwordForm()}
			</Panel>
		{/if}
		{#if needsTotp}
			<Panel title="Set up two-factor sign-in">
				<AuthenticatorSetup {setup} {qr} {qrError} bind:code busy={totpBusy} error={totpError} onstart={startTotp} onconfirm={confirmTotp} />
			</Panel>
		{:else if user?.two_factor_enabled}
			<Panel title="Two-factor sign-in">
				<p class="text-ok" role="status">Two-factor sign-in is on.</p>
			</Panel>
		{/if}
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
					{#if emailOn === true}
						<p class="muted">Open the link we sent to confirm it's yours.</p>
						{#if resent}
							<p class="muted">Sent another one.</p>
						{:else}
							<button class="btn self-start" onclick={resend}>Send the link again</button>
						{/if}
						{#if resendError}<p class="error" role="alert">{resendError}</p>{/if}
					{:else if emailOn === false}
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
					<p class="muted">Your request is pending.</p>
				{:else}
					<details>
						<summary class="cursor-pointer text-ink-dim hover:text-ink">Need a quota increase?</summary>
					<form class="mt-3 flex flex-col gap-3" onsubmit={askForMore}>
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
					</details>
				{/if}
			</Panel>
		{/if}

		{#if !needsPassword}
			<Panel title="Change password" open={false}>
				{@render passwordForm()}
			</Panel>
		{/if}
	</div>
</div>
