<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";

	let status = $state<"checking" | "done" | "failed">("checking");
	let error = $state("");

	$effect(() => {
		const token = page.url.searchParams.get("token") ?? "";
		api("/api/auth/verify-email", { body: { token } }).then(
			async () => {
				status = "done";
				if (session.current) await session.load();
			},
			(e) => {
				status = "failed";
				error = message(e);
			},
		);
	});
</script>

<h1>Confirm your email</h1>
{#if status === "checking"}
	<p class="muted">Checking the link…</p>
{:else if status === "done"}
	<p>Thanks, your email is confirmed.</p>
	<p><a href={session.current ? "/projects" : "/login"}>Continue</a></p>
{:else}
	<p class="error" role="alert">{error}</p>
	<p>Sign in and ask for a new link from your account page.</p>
{/if}
