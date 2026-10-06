/** Where to go after signing in: a path on this site, or the projects page. */
export function safeNext(target: string | null, origin: string): string {
	if (!target) return "/projects";
	try {
		const url = new URL(target, origin);
		if (url.origin !== origin || !target.startsWith("/")) return "/projects";
		return url.pathname + url.search + url.hash;
	} catch {
		return "/projects";
	}
}
