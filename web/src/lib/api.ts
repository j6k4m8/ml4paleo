/**
 * Calls to the ml4paleo API on this origin.
 *
 * The session lives in an HttpOnly cookie. State-changing requests also send
 * the CSRF token that every session response carries, which other sites
 * can't read.
 */

export class ApiError extends Error {
	constructor(
		public status: number,
		public detail: unknown,
	) {
		super(typeof detail === "string" ? detail : `HTTP ${status}`);
	}
}

let csrfToken: string | null = null;

const SAFE = new Set(["GET", "HEAD"]);

export async function api<T>(
	path: string,
	options: { method?: string; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
	const method = options.method ?? (options.body === undefined ? "GET" : "POST");
	const headers: Record<string, string> = {};
	if (options.body !== undefined) headers["Content-Type"] = "application/json";
	if (csrfToken && !SAFE.has(method)) headers["X-CSRF-Token"] = csrfToken;
	const response = await fetch(path, {
		method,
		headers,
		body: options.body === undefined ? undefined : JSON.stringify(options.body),
		signal: options.signal,
		credentials: "same-origin",
	});
	if (response.status === 204) return undefined as T;
	const isJson = response.headers.get("content-type")?.includes("json");
	const data = isJson ? await response.json() : await response.text();
	if (!response.ok) {
		throw new ApiError(response.status, isJson ? (data as { detail?: unknown }).detail : data);
	}
	if (isJson && data && typeof data === "object" && "csrf_token" in data) {
		csrfToken = (data as { csrf_token: string }).csrf_token;
	}
	return data as T;
}

/** A readable message for an error from `api`. */
export function message(error: unknown): string {
	if (error instanceof ApiError) {
		const detail = error.detail;
		if (typeof detail === "string") return detail;
		if (detail && typeof detail === "object" && "message" in detail) {
			return String((detail as { message: unknown }).message);
		}
		if (Array.isArray(detail)) return "Please check the form.";
		return `Something went wrong (HTTP ${error.status}).`;
	}
	return error instanceof Error ? error.message : String(error);
}
