import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CancelRequest, DecodeRequest } from "./decoding";

const BASE = "http://test/labels/";

const metadata = {
	zarr_format: 3,
	node_type: "array",
	shape: [64, 64, 64],
	data_type: "uint8",
	chunk_grid: { name: "regular", configuration: { chunk_shape: [64, 64, 64] } },
	chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
	fill_value: 0,
	codecs: [{ name: "bytes" }],
};

const load = (id: number, path = "class_1"): DecodeRequest => ({
	type: "load",
	id,
	url: BASE,
	path,
	region: [
		[0, 64],
		[0, 64],
		[0, 64],
	],
	derived: true,
});

describe("the decode worker", () => {
	const posted: { reply: Record<string, unknown>; transfer: unknown }[] = [];
	let send: (message: DecodeRequest | CancelRequest) => Promise<void>;
	// What the worker's postMessage throws for the next reply, if anything (a reply that can't be cloned, say).
	let refusal: Error | undefined;

	/** Answers the array's metadata, and each chunk as `chunk` says (by the request, to be able to hold one). */
	const serve = (chunk: (request: Request) => Promise<Response> | Response) => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async (request: Request) => (request.url.endsWith("/zarr.json") ? new Response(JSON.stringify(metadata)) : chunk(request))),
		);
	};

	beforeEach(async () => {
		posted.length = 0;
		refusal = undefined;
		const fake = {
			onmessage: null as unknown,
			postMessage: (reply: Record<string, unknown>, transfer: unknown) => {
				if (refusal) {
					const error = refusal;
					refusal = undefined;
					throw error;
				}
				posted.push({ reply, transfer });
			},
		};
		vi.stubGlobal("self", fake);
		vi.resetModules();
		await import("./decode.worker");
		send = (message) => (fake.onmessage as (event: { data: unknown }) => Promise<void>)({ data: message });
	});
	afterEach(() => vi.unstubAllGlobals());

	it("answers a load with the values read and what the server made them from, handing the buffer over", async () => {
		serve(() => new Response(new Uint8Array(64 ** 3).fill(5), { headers: { "x-pyramid-version": "9", "x-chunk-version": "3" } }));
		await send(load(7));
		expect(posted).toHaveLength(1);
		const { reply, transfer } = posted[0]!;
		expect(reply).toMatchObject({ id: 7, shape: [64, 64, 64], pyramid: 9, version: undefined });
		expect((reply.data as Uint8Array)[0]).toBe(5);
		expect(transfer).toEqual([(reply.data as Uint8Array).buffer]);
	});

	it("answers a 503 as it is, with its Retry-After, without asking again", async () => {
		const chunk = vi.fn(() => new Response("busy", { status: 503, headers: { "retry-after": "4" } }));
		serve(chunk);
		await send(load(1));
		expect(chunk).toHaveBeenCalledTimes(1);
		expect(posted[0]!.reply).toMatchObject({ id: 1, status: 503, retryAfter: "4" });
		expect(posted[0]!.transfer).toEqual([]);
	});

	it("stops a load it is told to cancel, and answers that it was", async () => {
		// The chunk never comes, and the read ends when its request is cancelled, as a fetch does.
		serve(
			(request) =>
				new Promise<Response>((_resolve, reject) => {
					request.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
				}),
		);
		const loading = send(load(3));
		await vi.waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
		await send({ type: "cancel", id: 3 });
		await loading;
		expect(posted).toHaveLength(1);
		expect(posted[0]!.reply).toMatchObject({ id: 3, error: expect.stringContaining("AbortError") });
	});

	it("answers a reply that can't be sent as a failed load, so the page's load ends", async () => {
		serve(() => new Response(new Uint8Array(64 ** 3).fill(5)));
		refusal = new DOMException("The reply could not be cloned", "DataCloneError");
		await send(load(4));
		expect(posted).toHaveLength(1);
		expect(posted[0]!.reply).toMatchObject({ id: 4, error: expect.stringContaining("DataCloneError") });
		expect(posted[0]!.reply).not.toHaveProperty("data");
		// The next load is answered as usual.
		await send(load(5));
		expect(posted[1]!.reply).toMatchObject({ id: 5, shape: [64, 64, 64] });
	});

	it("ignores a cancel for a load that isn't running", async () => {
		await send({ type: "cancel", id: 99 });
		expect(posted).toEqual([]);
	});
});
