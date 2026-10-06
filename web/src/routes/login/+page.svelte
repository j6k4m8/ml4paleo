<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { ApiError, api, message } from "#lib/api.ts";
	import { safeNext } from "#lib/navigation.ts";
	import { session } from "#lib/session.svelte.ts";
	import Card from "#lib/ui/Card.svelte";

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

<Card title="Sign in">
	<form class="flex flex-col gap-3" onsubmit={submit}>
		<label class="label">Username or email <input class="field" bind:value={username} autocomplete="username" required /></label>
		<label class="label">
			Password
			<input class="field" type="password" bind:value={password} autocomplete="current-password" required />
		</label>
		{#if needsCode}
			<label class="label">
				Two-factor code
				<input class="field" bind:value={code} inputmode="numeric" autocomplete="one-time-code" required />
			</label>
		{/if}
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		<button class="btn btn-primary h-7" disabled={busy}>Sign in</button>
	</form>
	{#snippet footer()}
		No account? <a href="/signup">Sign up</a>
		{#if emailEnabled}· <a href="/reset-password">Forgot your password?</a>{/if}
	{/snippet}
</Card>
