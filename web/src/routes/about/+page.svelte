<script lang="ts">
	import { PAPER_AUTHORS, PAPER_TITLE, PAPER_URL } from "#lib/publication.ts";
	import { crumbs } from "#lib/ui/crumbs.svelte.ts";
	$effect(() => crumbs.set([{ label: "About" }]));
	// Team and affiliations carried over from the existing project website.
	const leads = [
		{ name: "Melanie During", photo: "melanie-during.png", affiliation: "Uppsala University" },
		{ name: "Fredrik Gustafsson", photo: "fredrik-gustafsson.png", affiliation: "Uppsala University" },
		{ name: "Jordan Matelsky", photo: "jordan-matelsky.png", affiliation: "University of Pennsylvania · Johns Hopkins University Applied Physics Laboratory" },
	];
	const mentors = [
		{ name: "Per Ahlberg", photo: "per-ahlberg.png", affiliation: "Uppsala University" },
		{ name: "Thomas Schön", photo: "thomas-schon.png", affiliation: "Uppsala University" },
		{ name: "Konrad Kording", photo: "konrad-kording.png", affiliation: "University of Pennsylvania" },
		{ name: "Brock Wester", photo: "brock-wester.png", affiliation: "Johns Hopkins University Applied Physics Laboratory" },
	];
</script>

<svelte:head><title>About · ml4paleo</title><meta name="description" content="The science, code and people behind ml4paleo, a collaborative open-science project for fossil image segmentation." /></svelte:head>

<div class="mx-auto flex max-w-4xl flex-col gap-9 px-6 py-10 sm:px-10">
	<header>
		<h1 class="max-w-2xl text-3xl leading-tight">ml4paleo is a collaborative open-science project.</h1>
		<p class="mt-4 max-w-2xl text-base leading-relaxed text-ink-dim">Our goal is to develop and democratize simple and powerful machine learning models for paleontological research around the world.</p>
	</header>
	<section class="flex flex-col gap-3"><h2>The science</h2><p class="text-sm leading-relaxed text-ink-dim">Research made possible in part by ml4paleo:</p><div class="panel p-5 text-sm leading-relaxed"><a class="text-base font-medium" href={PAPER_URL} target="_blank" rel="noreferrer">{PAPER_TITLE}</a><p class="mt-2 text-ink-dim">{PAPER_AUTHORS} (2025). <i>Fossil Record</i> 28(1): 103–114.</p><a class="mt-2 inline-block break-all" href={PAPER_URL} target="_blank" rel="noreferrer">{PAPER_URL}</a></div></section>
	<section class="flex flex-col gap-3"><h2>The code</h2><p class="text-sm leading-relaxed text-ink-dim">All code is open source and licensed under the <a href="https://opensource.org/licenses/Apache-2.0" target="_blank" rel="noreferrer">Apache 2.0 license</a>. Explore the code, contribute improvements, or run your own installation at <a href="https://github.com/j6k4m8/ml4paleo" target="_blank" rel="noreferrer">j6k4m8/ml4paleo</a>.</p></section>
	<section class="flex flex-col gap-4">
		<h2>The team</h2>
		<p class="text-sm text-ink-dim">The project is co-led by:</p>
		<div class="grid gap-3 sm:grid-cols-3">
			{#each leads as person (person.name)}
				<div class="panel flex flex-col items-center p-5 text-center">
					<img src="/team/{person.photo}" alt={person.name} width="160" height="160" loading="lazy" decoding="async" class="mb-4 size-40 rounded-full object-cover" />
					<h3 class="text-sm font-medium">{person.name}</h3>
					<p class="mt-2 text-xs leading-relaxed text-ink-dim">{person.affiliation}</p>
				</div>
			{/each}
		</div>
		<h3 class="mt-2 text-sm text-ink-dim">With mentorship and guidance from</h3>
		<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
			{#each mentors as person (person.name)}
				<div class="flex flex-col items-center py-3 text-center">
					<img src="/team/{person.photo}" alt={person.name} width="120" height="120" loading="lazy" decoding="async" class="mb-3 size-30 rounded-full object-cover" />
					<h4 class="text-sm font-medium">{person.name}</h4>
					<p class="mt-1 text-xs leading-relaxed text-ink-dim">{person.affiliation}</p>
				</div>
			{/each}
		</div>
	</section>
	<footer class="flex flex-wrap gap-5 border-t border-line pt-5"><a href="/">Home</a><a href="/projects">Open ml4paleo</a></footer>
</div>
