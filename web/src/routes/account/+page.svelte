<script lang="ts">
	import { api, message } from "#lib/api.ts";
	import { session } from "#lib/session.svelte.ts";

	let current = $state("");
	let next = $state("");
	let passwordError = $state("");
	let passwordDone = $state(false);

	let setup: { secret: string; otpauth_uri: string } | null = $state(null);
	let code = $state("");
	let totpError = $state("");

	const steps = $derived(session.current?.required_steps ?? []);

	async function changePassword(event: SubmitEvent) {
		event.preventDefault();
		passwordError = "";
		try {
			await api("/api/auth/password", { body: { current_password: current, new_password: next } });
			passwordDone = true;
			current = next = "";
			await session.load();
		} catch (e) {
			passwordError = message(e);
		}
	}

	async function startTotp() {
		totpError = "";
		try {
			setup = await api("/api/auth/totp/setup", { method: "POST" });
		} catch (e) {
			totpError = message(e);
		}
	}

	async function confirmTotp(event: SubmitEvent) {
		event.preventDefault();
		totpError = "";
		try {
			await api("/api/auth/totp/confirm", { body: { code } });
			setup = null;
			await session.load();
		} catch (e) {
			totpError = message(e);
		}
	}
</script>

<h1>Account</h1>
{#if steps.length > 0}
	<p>Finish these steps to use ml4paleo.</p>
{/if}

<section>
	<h2>Password</h2>
	<form class="stack" onsubmit={changePassword}>
		<label>
			Current password
			<input type="password" bind:value={current} autocomplete="current-password" required />
		</label>
		<label>
			New password
			<input type="password" bind:value={next} autocomplete="new-password" minlength="12" required />
		</label>
		{#if passwordError}<p class="error" role="alert">{passwordError}</p>{/if}
		{#if passwordDone}<p>Password changed.</p>{/if}
		<button>Change password</button>
	</form>
</section>

{#if steps.includes("set_up_two_factor")}
	<section>
		<h2>Two-factor sign-in</h2>
		{#if setup}
			<p>Add this key to your authenticator app, then enter the code it shows.</p>
			<p><code>{setup.secret}</code></p>
			<p><a href={setup.otpauth_uri}>Open in an authenticator app</a></p>
			<form class="stack" onsubmit={confirmTotp}>
				<label>Code <input bind:value={code} inputmode="numeric" autocomplete="one-time-code" required /></label>
				<button>Turn on</button>
			</form>
		{:else}
			<button onclick={startTotp}>Set up two-factor sign-in</button>
		{/if}
		{#if totpError}<p class="error" role="alert">{totpError}</p>{/if}
	</section>
{/if}
