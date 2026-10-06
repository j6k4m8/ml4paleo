/** Keeping a page up to date with what collaborators change. */

// How often to pick up what others added, changed, or deleted.
export const REFRESH_MS = 30_000;

/**
 * Call `reload` every `every` milliseconds while the page is visible, and
 * whenever it comes back into view. Returns a function that stops.
 */
export function whileVisible(reload: () => void, every = REFRESH_MS): () => void {
	const maybe = () => {
		if (document.visibilityState === "visible") reload();
	};
	const timer = setInterval(maybe, every);
	document.addEventListener("visibilitychange", maybe);
	window.addEventListener("focus", maybe);
	return () => {
		clearInterval(timer);
		document.removeEventListener("visibilitychange", maybe);
		window.removeEventListener("focus", maybe);
	};
}
