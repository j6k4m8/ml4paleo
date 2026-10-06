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
	let config = $state({ signup_mode: "open", require_email: false, password_min_length: 12 });
	const invite = $derived(page.url.searchParams.get("invite") ?? undefined);

	$effect(() => {
		api<typeof config>("/api/auth/config").then(
			(loaded) => (config = loaded),
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
	{#if config.signup_mode === "invite" && !invite}
		<p class="rounded-sm border border-warn/40 bg-warn/10 p-2 text-warn">
			Signing up needs an invitation. Ask the people who run this site for one.
		</p>
	{/if}
	<form class="flex flex-col gap-3" onsubmit={submit}>
		<label class="label">Username <input class="field" bind:value={username} autocomplete="username" required /></label>
		<label class="label">
			{config.require_email ? "Email" : "Email (optional)"}
			<input class="field" type="email" bind:value={email} autocomplete="email" required={config.require_email} />
			{#if config.require_email}
				<span>Until you confirm it, your account has starter limits.</span>
			{/if}
		</label>
		<label class="label">
			Password (at least {config.password_min_length} characters)
			<input
				class="field"
				type="password"
				bind:value={password}
				autocomplete="new-password"
				minlength={config.password_min_length}
				required
			/>
		</label>
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		<button class="btn btn-primary h-7" disabled={busy}>Sign up</button>
	</form>
	{#snippet footer()}
		Have an account? <a href="/login">Sign in</a>
	{/snippet}
</Card>
