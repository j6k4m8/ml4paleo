<script lang="ts">
	let {
		setup, qr, qrError, code = $bindable(""), busy, error, onstart, onconfirm,
	}: {
		setup: { secret: string; otpauth_uri: string } | null;
		qr: string;
		qrError: boolean;
		code?: string;
		busy: boolean;
		error: string;
		onstart: () => void;
		onconfirm: (event: SubmitEvent) => void;
	} = $props();
</script>

{#if setup}
	<p>In your phone’s authenticator app, choose <strong>Add account</strong>, then <strong>Scan QR code</strong>.</p>
	{#if qr}
		<img class="w-64 max-w-full self-start rounded-sm" src={qr} width="256" height="256" alt="Scan this QR code with your authenticator app to add ml4paleo" />
	{:else if qrError}
		<p class="error" role="alert">The QR code couldn’t load. Use the setup key below instead.</p>
	{:else}
		<p role="status">Creating QR code…</p>
	{/if}
	<a class="self-start" href={setup.otpauth_uri}>On your phone? Open in your authenticator app</a>
	<details open={qrError}>
		<summary class="cursor-pointer text-ink-dim hover:text-ink">Can’t scan? Enter a setup key instead</summary>
		<div class="mt-2 flex flex-col gap-2">
			<p>Choose <strong>Enter setup key</strong> in your app. Name it <strong>ml4paleo</strong> and choose <strong>Time based</strong>.</p>
			<code class="select-all break-all rounded-sm border border-edge p-2 font-mono">{setup.secret}</code>
		</div>
	</details>
	<form class="mt-2 flex max-w-sm flex-col gap-3" onsubmit={onconfirm}>
		<label class="label">
			6-digit code from your authenticator app
			<input class="field w-40 font-mono" bind:value={code} inputmode="numeric" autocomplete="one-time-code" pattern={"[0-9]{6}"} maxlength="6" required disabled={busy} />
		</label>
		<button class="btn btn-primary self-start" disabled={busy}>{busy ? "Checking…" : "Enable two-factor sign-in"}</button>
	</form>
{:else}
	<p>Use an authenticator app on your phone, such as Google Authenticator or Microsoft Authenticator.</p>
	<button class="btn btn-primary self-start" onclick={onstart} disabled={busy}>{busy ? "Creating QR code…" : "Show QR code"}</button>
{/if}
{#if error}<p class="error" role="alert">{error}</p>{/if}
