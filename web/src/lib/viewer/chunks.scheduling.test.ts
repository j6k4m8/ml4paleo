import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Busy, type Chunk, ChunkStore, PATIENCE_MS, RETRY_GAP_MS } from "./chunks";

/** A seeded random number generator, so a failing run can be run again. */
function rng(seed: number) {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const chunk = (n = 10): Chunk => ({ data: new Uint8Array(n), shape: [1, 1, n] });
// Chunks the server takes long to make (`s`) and chunks it only reads (`f`).
const SLOW = Array.from({ length: 8 }, (_, i) => `s${i}`);
const FAST = Array.from({ length: 6 }, (_, i) => `f${i}`);
const IDS = [...SLOW, ...FAST];
const isSlow = (id: string) => id.startsWith("s");
const VIEWS = ["xy", "xz", "yz"];

interface Call {
	id: string;
	signal: AbortSignal;
	resolve: (c: Chunk) => void;
	reject: (e: unknown) => void;
	settled: boolean;
}

/** A loader whose loads end when the test says so, or when they are cancelled. */
function controlled(calls: Call[], onStart?: (call: Call) => void) {
	return (id: string, signal: AbortSignal) =>
		new Promise<Chunk>((resolve, reject) => {
			const call: Call = { id, signal, resolve, reject, settled: false };
			calls.push(call);
			signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
			onStart?.(call);
		});
}

describe("the store's scheduling, under random operations, against a model of when a load may start", () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	/**
	 * The model knows only what the store shows (`isLoading`, `peek`) and what the test did: the
	 * answers it gave, and what the views said they show. A place may stand free while a load
	 * waits only if that load can't start by the rules: a chunk answered busy that isn't due yet,
	 * or is held back by the gap between asks again, or a slow chunk past its places while a chunk
	 * that isn't slow waits (queued, or shown and not loaded or loading). A chunk is never asked
	 * again after its patience, and a chunk that ends busy does so with the answer it had last, no
	 * sooner than its patience allows. Some seeds make the server busy for every ask and let a lot of
	 * time pass between the test's turns, so that patience does run out.
	 */
	it("leaves no place idle while a load could start, keeps the slow places while another chunk waits, and asks again in time", async () => {
		let waiting = 0;
		let held = 0;
		// Chunks that ended busy: as they came up to be asked again after their patience, and as they were answered.
		let expired = 0;
		let gaveUp = 0;
		for (let seed = 1; seed <= 200; seed++) {
			const rand = rng(seed * 7919);
			// Every fifth seed: the server answers slow chunks busy nearly always, and much time passes between the test's turns.
			const stubborn = seed % 5 === 0;
			const concurrency = 2 + Math.floor(rand() * 3);
			const places = 1 + Math.floor(rand() * (concurrency - 1));
			const calls: Call[] = [];
			const errors: string[] = [];
			// What each view shows, as the test told the store.
			const shownBy = new Map<string, Set<string>>();
			// Chunks answered busy that wait to be asked again: id -> when they are due.
			const parked = new Map<string, number>();
			// When each chunk that is loading was first asked for, and the busy answer each had last.
			const began = new Map<string, number>();
			const lastBusy = new Map<string, Busy>();
			const seen = new WeakSet<object>();
			let retryFrom = 0;
			let store!: ChunkStore;
			const running = () => calls.filter((c) => !c.settled && !c.signal.aborted);
			/** Why a chunk that isn't slow is waiting for a place, if one is. */
			const fastWaiting = () => {
				const runningIds = new Set(running().map((c) => c.id));
				for (const id of FAST) if (store.isLoading(id) && !runningIds.has(id)) return `${id} queued`;
				for (const set of shownBy.values()) {
					for (const id of set) if (!isSlow(id) && !store.peek(id) && !store.isLoading(id)) return `${id} shown and not loaded`;
				}
				return "";
			};
			store = new ChunkStore(
				controlled(calls, (call) => {
					const { id } = call;
					const now = Date.now();
					if (parked.has(id)) {
						if (now < parked.get(id)!) errors.push(`${id} asked again at ${now}, before it was due at ${parked.get(id)}`);
						if (now < retryFrom) errors.push(`${id} asked again at ${now}, inside the gap (${retryFrom})`);
						if (now - began.get(id)! > PATIENCE_MS) errors.push(`${id} asked again at ${now}, ${now - began.get(id)!} ms after it was first asked for`);
						parked.delete(id);
						retryFrom = now + RETRY_GAP_MS;
					} else began.set(id, now);
					const run = running();
					if (run.length > concurrency) errors.push(`${run.length} loads of ${concurrency} places`);
					const slowRunning = run.filter((c) => isSlow(c.id)).length;
					if (isSlow(id) && slowRunning > places) {
						const why = fastWaiting();
						if (why) errors.push(`slow ${id} started as load ${slowRunning} of ${places} slow places while ${why}`);
					}
				}),
				1e9,
				concurrency,
				{ slow: isSlow, places },
			);
			const pick = <T>(xs: T[]) => xs[Math.floor(rand() * xs.length)]!;
			const answered: { id: string; due: number }[] = [];
			const answer = (call: Call) => {
				call.settled = true;
				const r = rand();
				if (isSlow(call.id) && r < (stubborn ? 0.97 : 0.45)) {
					const delay = 500 + Math.floor(rand() * 2500);
					const busy = new Busy(delay, null);
					lastBusy.set(call.id, busy);
					answered.push({ id: call.id, due: Date.now() + delay });
					call.reject(busy);
				} else if (r > (isSlow(call.id) ? 0.95 : 0.85)) call.reject(new Error("boom"));
				else call.resolve(chunk());
			};
			/**
			 * What a chunk's promise ended with: only what the test did (a cancel, or an answer that failed) or
			 * a busy answer, which ends it only with the last, and no sooner than the patience allows.
			 */
			const ended = (id: string) => (error: unknown) => {
				if (!(error instanceof Busy)) {
					const cancelled = error instanceof DOMException && error.name === "AbortError";
					if (!cancelled && (error as Error | undefined)?.message !== "boom") errors.push(`${id} ended with ${String(error)}`);
					return;
				}
				if (seen.has(error)) return;
				seen.add(error);
				const asked = began.get(id);
				const now = Date.now();
				if (asked === undefined) errors.push(`${id} ended busy at ${now}, and wasn't asked for`);
				else if (error !== lastBusy.get(id)) errors.push(`${id} ended busy with an answer that wasn't its last`);
				else if (now + error.delay - asked <= PATIENCE_MS) errors.push(`${id} ended busy at ${now - asked} ms of its patience, with a wait of ${error.delay} ms left`);
				else if (now - asked > PATIENCE_MS) expired++;
				else gaveUp++;
			};
			/** Let what settled run, then note which chunks answered busy now wait. */
			const settle = async () => {
				await vi.advanceTimersByTimeAsync(0);
				for (const b of answered.splice(0)) if (store.isLoading(b.id) && !running().some((c) => c.id === b.id)) parked.set(b.id, b.due);
				for (const id of [...parked.keys()]) if (!store.isLoading(id)) parked.delete(id);
				for (const id of [...began.keys()]) if (!store.isLoading(id)) began.delete(id);
			};
			/** A place standing free means nothing that waits could start, and no timer stays armed with nothing waiting. */
			const check = (label: string) => {
				const now = Date.now();
				const run = running();
				const runningIds = new Set(run.map((c) => c.id));
				if (vi.getTimerCount() > 0 && !IDS.some((id) => store.isLoading(id) && !runningIds.has(id))) errors.push(`${label}: a timer is armed with no load waiting`);
				if (run.length >= concurrency) return;
				const slowRunning = run.filter((c) => isSlow(c.id)).length;
				const fast = fastWaiting();
				for (const id of IDS) {
					if (!store.isLoading(id) || runningIds.has(id)) continue;
					waiting++;
					if (parked.has(id) && (now < parked.get(id)! || now < retryFrom)) continue;
					if (isSlow(id) && slowRunning >= places && fast) {
						held++;
						continue;
					}
					errors.push(`${label}: a place is free (${run.length} of ${concurrency}) and ${id} (${parked.has(id) ? "due, parked" : "queued"}) isn't started, at ${now}`);
					return;
				}
			};
			for (let step = 0; step < 300; step++) {
				const op = rand();
				const label = `seed ${seed} step ${step} op ${op.toFixed(2)}`;
				if (op < 0.12) {
					const id = pick(IDS);
					store.request(id).catch(ended(id));
				} else if (op < 0.16) {
					// A refresh takes the place of a load that waits: a new load, asked for at once.
					const id = pick(IDS);
					parked.delete(id);
					began.delete(id);
					store.refresh(id).catch(ended(id));
				} else if (op < 0.3) {
					const view = pick(VIEWS);
					const list = IDS.filter(() => rand() < 0.6).sort(() => rand() - 0.5);
					const r = rand();
					const shown = r < 0.3 ? undefined : r < 0.4 ? [] : list.filter(() => rand() < 0.5);
					// The store takes the new lists before it pumps, so the model does too.
					const kept = shown ?? list;
					if (kept.length > 0) shownBy.set(view, new Set(kept));
					else shownBy.delete(view);
					store.want(view, list, shown);
				} else if (op < 0.34) {
					store.keepOnly(new Set(IDS.filter(() => rand() < 0.6)));
				} else if (op < 0.37) {
					store.invalidate(pick(IDS));
				} else if (op < 0.4) {
					// A view is hidden or gone.
					const view = pick(VIEWS);
					shownBy.delete(view);
					store.want(view, []);
				} else if (op < 0.72) {
					const open = running();
					if (open.length > 0) answer(pick(open));
				} else {
					await vi.advanceTimersByTimeAsync(stubborn && rand() < 0.5 ? 5000 + Math.floor(rand() * 30_000) : Math.floor(rand() * 900));
				}
				await settle();
				check(label);
				if (errors.length > 0) break;
			}
			expect(errors, `seed ${seed}`).toEqual([]);
			// Drain: the views ask for nothing that isn't loaded, every load that runs is answered, and time passes.
			for (const view of VIEWS) {
				const list = IDS.filter((id) => store.isLoading(id) || store.peek(id));
				if (list.length > 0) shownBy.set(view, new Set(list));
				else shownBy.delete(view);
				store.want(view, list);
			}
			for (let i = 0; i < 2000; i++) {
				for (const c of running()) answer(c);
				await vi.advanceTimersByTimeAsync(250);
				await settle();
				check(`seed ${seed} drain ${i}`);
				if (!IDS.some((id) => store.isLoading(id))) break;
			}
			expect(errors, `seed ${seed} drain`).toEqual([]);
			expect(
				IDS.filter((id) => store.isLoading(id)),
				`seed ${seed}: still loading after the drain`,
			).toEqual([]);
			store.keepOnly(new Set());
			expect(vi.getTimerCount(), `seed ${seed}: timers left after cancelling everything`).toBe(0);
		}
		// The run did put waiting loads, and loads held back for a chunk that isn't slow, to the test.
		expect(waiting).toBeGreaterThan(10_000);
		expect(held).toBeGreaterThan(500);
		// Some chunks ended busy as they came up after their patience, some as they were answered.
		expect(expired).toBeGreaterThan(50);
		expect(gaveUp).toBeGreaterThan(5);
	}, 60_000);

	it("keeps a timer for as long as a chunk is waiting to be asked again, and none otherwise", async () => {
		const calls: Call[] = [];
		const store = new ChunkStore(controlled(calls), 1e9, 4, { slow: isSlow, places: 2 });
		for (const id of ["s0", "s1", "s2"]) store.request(id).catch(() => {});
		calls[0]!.reject(new Busy(5000, null));
		calls[1]!.reject(new Busy(20_000, null));
		await vi.advanceTimersByTimeAsync(0);
		expect(vi.getTimerCount()).toBe(1);
		store.keepOnly(new Set(["s1", "s2"]));
		expect(vi.getTimerCount(), "s1 still waits, s2 runs").toBe(1);
		store.invalidate("s1");
		expect(vi.getTimerCount(), "only s2 is left, and it runs").toBe(0);
		store.keepOnly(new Set());
		expect(vi.getTimerCount()).toBe(0);
		// A chunk that waits later gets its timer again, and is asked for when it is due.
		store.request("s3").catch(() => {});
		await vi.advanceTimersByTimeAsync(0);
		const call = calls.find((c) => c.id === "s3" && !c.settled)!;
		call.settled = true;
		call.reject(new Busy(1000, null));
		await vi.advanceTimersByTimeAsync(0);
		expect(vi.getTimerCount()).toBe(1);
		await vi.advanceTimersByTimeAsync(1100);
		expect(calls.filter((c) => c.id === "s3")).toHaveLength(2);
	});
});
