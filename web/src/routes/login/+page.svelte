<script lang="ts">
	import { goto } from "$app/navigation";
	import { page } from "$app/state";
	import { ApiError, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";

	let username = $state("");
	let password = $state("");
	let code = $state("");
	let needsCode = $state(false);
	let error = $state("");
	let busy = $state(false);

	function next(): string {
		const target = page.url.searchParams.get("next") ?? "";
		// Only paths on this site.
		return target.startsWith("/") && !target.startsWith("//") ? target : "/projects";
	}

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		busy = true;
		error = "";
		try {
			await session.login(username, password, needsCode ? code : undefined);
			goto(next());
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
	<p class="muted">No account? <a href="/signup">Sign up</a></p>
</form>
