/**
 * The project's Neuroglancer tab: what it shows, and the links it opens.
 *
 * The server serves Neuroglancer on this site, at `/neuroglancer/`, and builds
 * the link to a project's view (`GET /api/projects/<id>/neuroglancer`). The
 * view is read-only: painting stays in the annotator.
 */

/** Where this site's Neuroglancer lives; the server's links go there. */
export const NEUROGLANCER_PATH = "/neuroglancer/";

export type NeuroglancerView =
	| { kind: "loading" }
	/** The link to show in the frame. */
	| { kind: "ready"; url: string }
	/** This server has no Neuroglancer. */
	| { kind: "unavailable" }
	/** The project has no image to show yet. */
	| { kind: "no-image" }
	| { kind: "error"; message: string };

/** Whether `url` is a link into this site's Neuroglancer (a path on this site, with the view after a `#`). */
export function isNeuroglancerLink(url: unknown): url is string {
	if (typeof url !== "string" || !url.startsWith(NEUROGLANCER_PATH)) return false;
	try {
		const base = new URL("https://site.invalid");
		const link = new URL(url, base);
		return link.origin === base.origin && link.pathname === NEUROGLANCER_PATH;
	} catch {
		return false;
	}
}

/**
 * What the server's answer comes to: a link to show, or no Neuroglancer on
 * this server (`url` is null). A link that doesn't go to this site's
 * Neuroglancer isn't framed, whatever the answer says.
 */
export function answerView(answer: { url?: unknown } | null | undefined): NeuroglancerView {
	const url = answer?.url;
	if (url === null || url === undefined) return { kind: "unavailable" };
	if (isNeuroglancerLink(url)) return { kind: "ready", url };
	return { kind: "error", message: "The server's link to Neuroglancer isn't one this page can show." };
}

/**
 * Where "Open in its own window" goes: the view the frame shows now (Neuroglancer keeps
 * its state in the address as people look around), if that is a page of this site's
 * Neuroglancer, else the link the tab began with.
 */
export function ownWindowHref(frameHref: string | null | undefined, origin: string, began: string): string {
	try {
		const now = new URL(frameHref ?? "");
		if (now.origin === origin && now.pathname === NEUROGLANCER_PATH) return now.pathname + now.search + now.hash;
	} catch {
		// Not an address (the frame hasn't loaded, or is gone).
	}
	return began;
}
