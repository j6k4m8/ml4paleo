/** Only transport/server failures are retryable, not bad data or denied access. */
export class LoadError extends Error {
	constructor(
		message: string,
		readonly status?: number,
		readonly retryAfter?: string,
		readonly transient = status !== undefined && [408, 429, 500, 502, 503, 504].includes(status),
	) {
		super(message);
		this.name = "LoadError";
	}
}

export const LOAD_RETRIES = 3;

/** Back off with jitter, never earlier than the server's Retry-After. */
export function loadRetryDelay(error: LoadError, attempt: number): number {
	const value = error.retryAfter?.trim();
	const seconds = value ? Number(value) : NaN;
	const serverWait = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(value ?? "") - Date.now();
	return Math.max(1000 * 2 ** (attempt - 1), Number.isFinite(serverWait) ? serverWait : 0) * (1 + Math.random() * 0.5);
}
