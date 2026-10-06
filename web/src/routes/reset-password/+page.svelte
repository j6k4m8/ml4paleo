<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import Card from "#lib/ui/Card.svelte";

	const token = $derived(page.url.searchParams.get("token"));
	let email = $state("");
	let password = $state("");
	let sent = $state(false);
	let error = $state("");
	let busy = $state(false);

	async function request(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await api("/api/auth/password-reset/request", { body: { email } });
			sent = true;
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}

	async function confirm(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await api("/api/auth/password-reset/confirm", { body: { token, new_password: password } });
			await goto("/login", { replace: true });
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}
</script>

<Card title="Reset your password">
	{#if token}
		<form class="flex flex-col gap-3" onsubmit={confirm}>
			<label class="label">
				New password (at least 12 characters)
				<input class="field" type="password" bind:value={password} autocomplete="new-password" minlength="12" required />
			</label>
			{#if error}<p class="error" role="alert">{error}</p>{/if}
			<button class="btn btn-primary h-7" disabled={busy}>Set password</button>
		</form>
	{:else if sent}
		<p>If that email belongs to an account, we sent it a link. It works for an hour.</p>
	{:else}
		<form class="flex flex-col gap-3" onsubmit={request}>
			<label class="label">Email <input class="field" type="email" bind:value={email} autocomplete="email" required /></label>
			{#if error}<p class="error" role="alert">{error}</p>{/if}
			<button class="btn btn-primary h-7" disabled={busy}>Send a link</button>
		</form>
	{/if}
</Card>
