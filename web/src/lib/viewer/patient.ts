/**
 * Asking again for what a server is still making. The zoomed-out levels of
 * the labels are worked out when asked for, and a chunk too big to make in
 * one request is answered 503 with `Retry-After`, the work done so far kept,
 * so asking again gets further. zarrita takes any answer that isn't a chunk
 * or a 404 for an error, so the decode workers ask again themselves.
 */

/** The longest a chunk is waited for, counted from the first ask, before its 503 is passed on. */
export const PATIENCE_MS = 120_000;
// What a 503 without a usable Retry-After is waited, and the least and the
// most any is, in seconds.
const DEFAULT_WAIT = 2;
const LEAST_WAIT = 0.5;
const LONGEST_WAIT = 30;
// Each wait is up to this much longer at random, so chunks asked for
// together don't all come back together.
const SPREAD = 0.5;

/**
 * How long to wait, in milliseconds, after a 503 with this `Retry-After`
 * (seconds, or a date): what it says and a little more, by `random`, from 0
 * to 1.
 */
export function retryDelay(header: string | null, random: number, now: number = Date.now()): number {
	const given = header?.trim() ? Number(header) : Number.NaN;
	const seconds = Number.isFinite(given) ? given : (Date.parse(header ?? "") - now) / 1000;
	const wait = Math.min(LONGEST_WAIT, Math.max(LEAST_WAIT, Number.isFinite(seconds) ? seconds : DEFAULT_WAIT));
	return wait * 1000 * (1 + SPREAD * random);
}

/** Resolves after `ms`, or rejects at once, as a fetch does, if `signal` aborts first. */
function pause(ms: number, signal: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		const abort = () => {
			clearTimeout(timer);
			reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
		};
		const timer = setTimeout(() => {
			signal.removeEventListener("abort", abort);
			resolve();
		}, ms);
		if (signal.aborted) return abort();
		signal.addEventListener("abort", abort, { once: true });
	});
}

/**
 * A fetch that asks again after a 503, once the `Retry-After` and a little
 * more have passed, until the chunk comes or `patience` has gone by since the
 * first ask, when the 503 is passed on as it is. Cancelling the request ends
 * the wait. Any other answer is passed on at once.
 */
export function patientFetch(
	fetcher: (request: Request) => Promise<Response>,
	{ patience = PATIENCE_MS, random = Math.random }: { patience?: number; random?: () => number } = {},
): (request: Request) => Promise<Response> {
	return async (request) => {
		const started = Date.now();
		for (;;) {
			const response = await fetcher(request);
			if (response.status !== 503) return response;
			const wait = retryDelay(response.headers.get("retry-after"), random());
			if (Date.now() - started + wait > patience) return response;
			// Nothing reads this one, so let its connection go.
			void response.body?.cancel().catch(() => {});
			await pause(wait, request.signal);
		}
	};
}
