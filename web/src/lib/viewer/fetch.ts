/** Includes response-body reads, not just the arrival of HTTP headers.
 * Shared metadata/index reads have their own deadline, independent of the
 * first chunk's lifetime. A cancelled view must not cancel another view's index.
 */
export const READ_TIMEOUT_MS = 20_000;

export function timedFetch(fetcher: (request: Request) => Promise<Response> = (request) => fetch(request)) {
	return (request: Request): Promise<Response> => fetcher(new Request(request, {
		signal: AbortSignal.any([request.signal, AbortSignal.timeout(READ_TIMEOUT_MS)]),
	}));
}
