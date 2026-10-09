<script lang="ts">
	import "../app.css";
	import ChevronRight from "@lucide/svelte/icons/chevron-right";
	import LogOut from "@lucide/svelte/icons/log-out";
	import MailWarning from "@lucide/svelte/icons/mail-warning";
	import User from "@lucide/svelte/icons/user";
	import { afterNavigate, goto } from "$app/navigation";
	import { page } from "$app/state";
	import { session } from "#lib/session.svelte.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";

	let { children } = $props();

	const PUBLIC = ["/login", "/signup", "/verify-email", "/reset-password"];
	// Pages that help finish the required account steps.
	const SETUP = ["/account", "/verify-email"];
	// The annotator and the Neuroglancer tab fill the window; other pages scroll.
	const workspace = $derived(page.url.pathname.endsWith("/annotate") || page.url.pathname.endsWith("/neuroglancer"));
	let menu = $state(false);
	let account: HTMLElement | undefined = $state();
	let accountButton: HTMLButtonElement | undefined = $state();

	afterNavigate(() => (menu = false));

	function menuKey(event: KeyboardEvent) {
		if (!menu || event.key !== "Escape") return;
		// Before anything else on the page sees it, such as the annotator.
		event.stopPropagation();
		menu = false;
		accountButton?.focus();
	}

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

	// Each page names itself; until it does, show nothing stale. This runs
	// before the new page's effects, so it doesn't wipe what they set.
	$effect.pre(() => {
		void page.url.pathname;
		crumbs.set([]);
	});

	async function logout() {
		menu = false;
		await session.logout();
		goto("/login");
	}
</script>

<svelte:window
	onkeydowncapture={menuKey}
	onpointerdown={(event) => {
		if (menu && !account?.contains(event.target as Node)) menu = false;
	}}
/>

<div class="flex h-full flex-col">
	<header class="flex h-9 shrink-0 items-center gap-3 border-b border-edge bg-chrome px-3">
		<a href="/projects" class="flex items-center gap-2 text-ink no-underline hover:no-underline" aria-label="ml4paleo home">
			<span class="grid size-5 place-items-center rounded-[3px] bg-accent-fill text-[9px] font-bold tracking-tight text-white">m4</span>
			<span class="text-xs font-semibold">ml4paleo</span>
		</a>
		{#if crumbs.items.length > 0}
			<nav aria-label="Breadcrumbs" class="flex min-w-0 items-center gap-1 text-ink-dim">
				{#each crumbs.items as crumb, index (index)}
					<ChevronRight size={12} class="shrink-0 text-ink-faint" />
					{#if crumb.href}
						<a href={crumb.href} class="truncate text-ink-dim hover:text-ink hover:no-underline">{crumb.label}</a>
					{:else}
						<span class="truncate text-ink">{crumb.label}</span>
					{/if}
				{/each}
			</nav>
		{/if}
		{#if session.current}
			<!-- Tabbing or clicking away closes the menu. -->
			<div
				class="relative ml-auto"
				bind:this={account}
				onfocusout={(event) => {
					if (event.relatedTarget instanceof Node && !account?.contains(event.relatedTarget)) menu = false;
				}}
			>
				<button
					class="btn btn-ghost"
					bind:this={accountButton}
					onclick={() => (menu = !menu)}
					aria-expanded={menu}
					aria-controls="account-menu"
				>
					<User size={14} />
					{session.current.user.username}
				</button>
				{#if menu}
					<div
						id="account-menu"
						class="absolute top-7 right-0 z-30 flex w-44 flex-col rounded-sm border border-edge bg-panel py-1 shadow-lg shadow-black/40"
					>
						<a href="/account" class="px-3 py-1.5 text-ink hover:bg-accent-fill hover:text-white hover:no-underline" onclick={() => (menu = false)}>
							Account
						</a>
						{#if session.current.user.is_admin}
							<a href="/admin" class="px-3 py-1.5 text-ink hover:bg-accent-fill hover:text-white hover:no-underline" onclick={() => (menu = false)}>
								Administration
							</a>
						{/if}
						<button class="flex items-center gap-2 px-3 py-1.5 text-left hover:bg-accent-fill hover:text-white" onclick={logout}>
							<LogOut size={13} /> Sign out
						</button>
					</div>
				{/if}
			</div>
		{/if}
	</header>

	{#if session.current?.starter_limits && !workspace && !SETUP.includes(page.url.pathname)}
		<div class="flex h-7 shrink-0 items-center gap-2 border-b border-warn/30 bg-warn/10 px-3 text-2xs text-warn" role="status">
			<MailWarning size={13} />
			Your account has starter limits until it has a confirmed email address.
			<a href="/account" class="text-warn underline">Account</a>
		</div>
	{/if}

	<main class={workspace ? "min-h-0 flex-1" : "min-h-0 flex-1 overflow-auto bg-pasteboard"}>
		{#if session.loaded}
			{@render children()}
		{/if}
	</main>
</div>
