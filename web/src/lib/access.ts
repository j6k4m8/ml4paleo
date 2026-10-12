const PUBLIC = new Set(["/", "/about", "/login", "/signup", "/verify-email", "/reset-password"]);
const SETUP = new Set(["/account", "/verify-email"]);

/** Public information stays readable without signing in or completing setup. */
export function accessRedirect(path: string, search: string, signedIn: boolean, steps: string[]): string | null {
	if (!signedIn && !PUBLIC.has(path)) return `/login?next=${encodeURIComponent(path + search)}`;
	if (signedIn && steps.length && !SETUP.has(path) && path !== "/about" && path !== "/") {
		return `/account?next=${encodeURIComponent(path + search)}`;
	}
	return null;
}
