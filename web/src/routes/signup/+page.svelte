<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import Card from "#lib/ui/Card.svelte";

	let username = $state("");
	let email = $state("");
	let password = $state("");
	let error = $state("");
	let busy = $state(false);
	let mode = $state("open");
	const invite = $derived(page.url.searchParams.get("invite") ?? undefined);

	$effect(() => {
		api<{ signup_mode: string }>("/api/auth/config").then(
			(config) => (mode = config.signup_mode),
			() => {},
		);
	});

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await session.signup(username, password, email, invite);
			goto("/projects");
		} catch (e) {
			error = message(e);
		} finally {
			busy = false;
		}
	}
</script>

<Card title="Create an account">
	{#if mode === "invite" && !invite}
		<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">
			Signing up needs an invitation. Ask the people who run this site for one.
		</p>
	{/if}
	<form class="flex flex-col gap-3" onsubmit={submit}>
		<label class="label">Username <input class="field" bind:value={username} autocomplete="username" required /></label>
		<label class="label">Email (optional) <input class="field" type="email" bind:value={email} autocomplete="email" /></label>
		<label class="label">
			Password (at least 12 characters)
			<input class="field" type="password" bind:value={password} autocomplete="new-password" minlength="12" required />
		</label>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		<button class="btn btn-primary h-7" disabled={busy}>Sign up</button>
	</form>
	{#snippet footer()}
		Have an account? <a href="/login">Sign in</a>
	{/snippet}
</Card>
