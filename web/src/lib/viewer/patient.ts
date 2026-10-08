/**
 * How long to wait for what a server is still making. The zoomed-out levels
 * of the labels are worked out when asked for, and a chunk too big to make in
 * one request is answered 503 with `Retry-After`, the work done so far kept,
 * so asking again gets further. zarrita takes any answer that isn't a chunk
 * or a 404 for an error, so the decode workers pass the 503 on, and the
 * chunk store (see `Busy`) asks again once `retryDelay` has gone by.
 */

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
