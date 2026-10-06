<script lang="ts">
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	import Panel from "#lib/ui/Panel.svelte";

	let current = $state("");
	let next = $state("");
	let passwordError = $state("");
	let passwordDone = $state(false);

	let setup: { secret: string; otpauth_uri: string } | null = $state(null);
	let code = $state("");
	let totpError = $state("");

	const steps = $derived(session.current?.required_steps ?? []);
	$effect(() => crumbs.set([{ label: "Account" }]));
	let resent = $state(false);
	let resendError = $state("");

	async function resend() {
		resendError = "";
		try {
			await api("/api/auth/verify-email/resend", { method: "POST" });
			resent = true;
		} catch (e) {
			resendError = message(e);
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
</script>

<div class="mx-auto flex max-w-xl flex-col gap-3 p-6">
	<h1>Account</h1>
	{#if steps.length > 0}
		<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">Finish these steps to use ml4paleo.</p>
	{/if}

	{#if steps.includes("verify_email")}
		<Panel title="Email">
			<p>Open the link we sent to {session.current?.user.email} to confirm it's yours.</p>
			{#if resent}
				<p class="muted">Sent another one.</p>
			{:else}
				<button class="btn self-start" onclick={resend}>Send the link again</button>
			{/if}
			{#if resendError}<p class="error" role="alert">{resendError}</p>{/if}
		</Panel>
	{/if}

	<div class="panel overflow-hidden">
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
