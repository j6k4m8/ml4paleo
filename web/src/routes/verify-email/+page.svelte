<script lang="ts">
	import { page } from "$app/state";
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";
	import Card from "#lib/ui/Card.svelte";

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

<Card title="Confirm your email">
	{#if status === "checking"}
		<p class="muted">Checking the link…</p>
	{:else if status === "done"}
		<p>Thanks, your email is confirmed.</p>
		<a class="btn btn-primary h-7 hover:no-underline" href={session.current ? "/projects" : "/login"}>Continue</a>
	{:else}
		<p class="error" role="alert">{error}</p>
		<p class="muted">Sign in and ask for a new link from your account page.</p>
	{/if}
</Card>
