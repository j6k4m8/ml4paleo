<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";

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

<h1>Sign up</h1>
{#if mode === "invite" && !invite}
	<p>Signing up needs an invitation. Ask the people who run this site for one.</p>
{/if}
<form class="stack" onsubmit={submit}>
	<label>Username <input bind:value={username} autocomplete="username" required /></label>
	<label>Email (optional) <input type="email" bind:value={email} autocomplete="email" /></label>
	<label>
		Password (at least 12 characters)
		<input type="password" bind:value={password} autocomplete="new-password" minlength="12" required />
	</label>
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	<button disabled={busy}>Sign up</button>
	<p class="muted">Have an account? <a href="/login">Sign in</a></p>
</form>
