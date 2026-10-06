<script lang="ts">
	import "../app.css";
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { session } from "#lib/session.svelte.ts";

	let { children } = $props();

	const PUBLIC = ["/login", "/signup", "/verify-email", "/reset-password"];
	// Pages that help finish the required account steps.
	const SETUP = ["/account", "/verify-email"];

	$effect(() => {
		if (!session.loaded) {
			session.load();
			return;
		}
		const path = page.url.pathname;
		const steps = session.current?.required_steps ?? [];
		if (!session.current && !PUBLIC.includes(path)) {
			goto(`/login?next=${encodeURIComponent(path + page.url.search)}`, { replace: true });
		} else if (session.current && steps.length > 0 && !SETUP.includes(path)) {
			goto("/account", { replace: true });
		}
	});

	async function logout() {
		await session.logout();
		goto("/login");
	}
</script>

<header>
	<a class="brand" href="/projects">ml4paleo</a>
	{#if session.current}
		<span class="muted">{session.current.user.username}</span>
		<a href="/account">Account</a>
		<button class="secondary" onclick={logout}>Sign out</button>
	{/if}
</header>

<main>
	{#if session.loaded}
		{@render children()}
	{/if}
</main>

<style>
	header {
		display: flex;
		gap: 1rem;
		align-items: center;
		padding: 0.5rem 1rem;
		border-bottom: 1px solid var(--line);
	}
	.brand {
		font-weight: 600;
		text-decoration: none;
		color: var(--fg);
		margin-right: auto;
	}
	main {
		padding: 1rem;
		height: calc(100vh - 3rem);
		overflow: auto;
	}
</style>
