<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { safeNext } from "#lib/navigation.ts";
	import { session } from "#lib/session.svelte.ts";

	let username = $state("");
	let password = $state("");
	let code = $state("");
	let needsCode = $state(false);
	let error = $state("");
	let busy = $state(false);

	let emailEnabled = $state(false);

	$effect(() => {
		api<{ email_enabled: boolean }>("/api/auth/config").then(
			(config) => (emailEnabled = config.email_enabled),
			() => {},
		);
	});

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await session.login(username, password, needsCode ? code : undefined);
			await goto(safeNext(page.url.searchParams.get("next"), page.url.origin));
		} catch (e) {
			if (e instanceof ApiError && e.detail === "totp_required") needsCode = true;
			else error = message(e);
		} finally {
			busy = false;
		}
	}
</script>

<h1>Sign in</h1>
<form class="stack" onsubmit={submit}>
	<label>Username or email <input bind:value={username} autocomplete="username" required /></label>
	<label>
		Password
		<input type="password" bind:value={password} autocomplete="current-password" required />
	</label>
	{#if needsCode}
		<label>
			Two-factor code
			<input bind:value={code} inputmode="numeric" autocomplete="one-time-code" required />
		</label>
	{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	<button disabled={busy}>Sign in</button>
	<p class="muted">
		No account? <a href="/signup">Sign up</a>
		{#if emailEnabled}· <a href="/reset-password">Forgot your password?</a>{/if}
	</p>
</form>
