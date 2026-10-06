/**
 * Jobs from the ml4paleo v1 app this one replaced. v1 kept the jobs each
 * browser had opened in `localStorage["jobs"]`, as `[{id, name}]`, and this
 * app runs on the same site, so it can read them.
 */

export interface V1Job {
	id: string;
	name: string;
}

const JOB_ID = /^[0-9A-F]{6}$/;

/**
 * The v1 job id in a pasted id or link, or null. v1 linked to a job's page
 * (`…/job/ab12cd`), the pages under it (`…/job/ab12cd/models`), and its
 * annotators (`…/job/annotate/ab12cd`, `…/job/annotate-v2/ab12cd`).
 */
export function parseJobId(text: string): string | null {
	const match = /(?:^|\/job\/(?:annotate(?:-v2)?\/)?)([0-9a-f]{6})(?:[/?#]|$)/i.exec(text.trim());
	return match?.[1] ? match[1].toUpperCase() : null;
}

/** The v1 jobs this browser remembers, once each, in the order it saved them. */
export function rememberedJobs(storage: Pick<Storage, "getItem"> | null): V1Job[] {
	let raw: unknown;
	try {
		raw = JSON.parse(storage?.getItem("jobs") ?? "[]");
	} catch {
		return [];
	}
	if (!Array.isArray(raw)) return [];
	const seen = new Set<string>();
	const jobs: V1Job[] = [];
	for (const item of raw) {
		const id = typeof item?.id === "string" ? item.id.toUpperCase() : "";
		if (!JOB_ID.test(id) || seen.has(id)) continue;
		seen.add(id);
		jobs.push({ id, name: typeof item.name === "string" ? item.name : "" });
	}
	return jobs;
}
