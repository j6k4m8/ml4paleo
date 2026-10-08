/**
 * A project's label history: every edit, undo, and redo, who made it, and
 * where its labels came from, in plain words.
 */

import type { Box } from "./rois.svelte";
import { annotatorHref } from "./viewer/link";

/** Where accepted labels came from. */
export interface Accepted {
	/** A prediction of the whole image, or one person's proposal for one ROI. */
	kind: "prediction" | "proposal";
	model_id: string | null;
	model_name: string | null;
	/** Set for a prediction brought over from v1, which no model made. */
	v1_job_id: string | null;
	/** The ROI they went into; none for labels accepted in a view, whose op's box says where. */
	roi_id: string | null;
}

/** One op, as `GET /api/projects/<id>/labels/ops` gives it. */
export interface Entry {
	seq: number;
	kind: "edit" | "undo" | "redo";
	user_id: string | null;
	/** Who made it; null for a job's edits (see `job_kind`). */
	username: string | null;
	job_kind: string | null;
	/** `ml4paleo.labels.Source`; an undo or redo has its edit's. */
	source: number;
	tool: Record<string, unknown>;
	bbox: Box;
	/** Whether an edit counts: it hasn't been undone, or was redone since. */
	live: boolean;
	accepted: Accepted | null;
	/** For an undo or redo: the edit it acts on, as that is now. */
	target_seq: number | null;
	target: Entry | null;
	created_at: string;
}

/** Where labels can come from (`ml4paleo.labels.Source`), for choosing. */
export const SOURCES: [number, string][] = [
	[1, "By hand"],
	[2, "Accepted from a model"],
	[3, "Click-to-segment"],
	[4, "Propagated"],
	[5, "Imported"],
];

const MODEL_VERIFIED = 2;

/** The tools people's edits name, in words. */
const TOOLS = new Map([
	["brush", "Brush stroke"],
	["eraser", "Eraser stroke"],
	["polygon", "Polygon"],
	["polygon-erase", "Polygon erase"],
]);

/** Jobs that write labels: who they are, and what each of their edits is. */
const JOBS = new Map([
	["v1.labels", { who: "v1 import", what: "v1 annotation" }],
	["labels.import", { who: "Label import", what: "Labels from a file" }],
]);

export type Action = "brush" | "eraser" | "polygon" | "polygon-erase" | "accept" | "import" | "edit" | "undo" | "redo";

/** The kind of thing an op did. */
export function action(entry: Entry): Action {
	if (entry.kind !== "edit") return entry.kind;
	// Only the server says labels were accepted, or that a job made them.
	if (entry.accepted) return "accept";
	if (entry.job_kind) return JOBS.has(entry.job_kind) ? "import" : "edit";
	const name = entry.tool.name;
	return typeof name === "string" && TOOLS.has(name) ? (name as Action) : "edit";
}

/** What an op did, in a few words. */
export function what(entry: Entry): string {
	const done = action(entry);
	if (done === "undo") return `Undid #${entry.target_seq}`;
	if (done === "redo") return `Redid #${entry.target_seq}`;
	if (done === "accept") return `Accepted ${entry.accepted?.kind ?? "prediction"}`;
	if (done === "import") {
		// A label import's job names the file the person uploaded.
		const file = entry.job_kind === "labels.import" ? entry.tool.file : undefined;
		if (typeof file === "string" && file) return `From ${file}`;
		return JOBS.get(entry.job_kind ?? "")?.what ?? "Edit";
	}
	return TOOLS.get(done) ?? "Edit";
}

/** Who made an op: a member, or the job that did (an import, say). */
export function who(entry: Entry): string {
	if (entry.username) return entry.username;
	if (entry.job_kind) return JOBS.get(entry.job_kind)?.who ?? "A job";
	return "Someone";
}

/** Where an op's labels came from; an undo or redo's are its edit's. */
export function origin(entry: Entry): string {
	const edit = entry.target ?? entry;
	if (edit.source === MODEL_VERIFIED) {
		const from = edit.accepted;
		if (from?.model_name) return `From ${from.model_name}`;
		if (from?.v1_job_id) return `From v1 job ${from.v1_job_id}`;
		return "From a model";
	}
	return SOURCES.find(([value]) => value === edit.source)?.[1] ?? "Unknown";
}

/** A box's size (x × y × z) and its corner nearest the origin. */
export function place(bbox: Box): { size: string; x: number; y: number; z: number } {
	const [z0, y0, x0, z1, y1, x1] = bbox;
	return { size: `${x1 - x0} × ${y1 - y0} × ${z1 - z0}`, x: x0, y: y0, z: z0 };
}

/** When something happened, counted back from `now` while that's short. */
export function ago(iso: string, now: number): string {
	const minutes = Math.floor((now - Date.parse(iso)) / 60_000);
	if (minutes < 1) return "just now";
	if (minutes < 60) return `${minutes} min ago`;
	if (minutes < 24 * 60) return `${Math.floor(minutes / 60)} h ago`;
	return new Date(iso).toLocaleString();
}

/** The annotator on an op's box, and on its ROI if its labels were accepted in one. */
export function link(projectId: string, entry: Entry): string {
	const edit = entry.target ?? entry;
	return annotatorHref(projectId, { roi: edit.accepted?.roi_id, box: entry.bbox });
}

/** Undoing or redoing an edit, by its seq. */
export interface Toggle {
	seq: number;
	action: "undo" | "redo";
}

/**
 * The Undo or Redo button of each entry that has one, by the entry's seq.
 * An edit can be undone while live and redone while undone. The newest undo
 * or redo of an edit listed can be reversed while the edit is as it left it.
 */
export function toggles(entries: Entry[]): Map<number, Toggle> {
	const buttons = new Map<number, Toggle>();
	// Edits whose newest undo or redo has been passed, going back in time.
	const passed = new Set<number>();
	for (const entry of entries) {
		const target = entry.target;
		if (entry.kind === "edit") {
			buttons.set(entry.seq, { seq: entry.seq, action: entry.live ? "undo" : "redo" });
		} else if (target && !passed.has(target.seq)) {
			passed.add(target.seq);
			if (entry.kind === "undo" && !target.live) buttons.set(entry.seq, { seq: target.seq, action: "redo" });
			if (entry.kind === "redo" && target.live) buttons.set(entry.seq, { seq: target.seq, action: "undo" });
		}
	}
	return buttons;
}

/**
 * Every entry `page(before)` gives, newest first: it's asked for the
 * entries before the last one it gave until it gives fewer than `size`.
 */
export async function everything(page: (before: number | null) => Promise<Entry[]>, size: number): Promise<Entry[]> {
	const all: Entry[] = [];
	for (let before: number | null = null; ; ) {
		const batch = await page(before);
		all.push(...batch);
		const last = batch.at(-1);
		if (batch.length < size || !last) return all;
		before = last.seq;
	}
}

/** Entries, newest first, and whether older ones may be left to load. */
export interface Listing {
	entries: Entry[];
	more: boolean;
}

/**
 * The listing after a reload: `fresh` is every entry from the newest back
 * to `oldest`, the oldest one shown when the reload started (or, if none
 * was, the newest `size`). Older ones loaded meanwhile stay below it.
 */
export function refreshed(fresh: Entry[], oldest: number | undefined, shown: Listing, size: number): Listing {
	if (oldest === undefined) return { entries: fresh, more: fresh.length === size };
	return { entries: [...fresh, ...shown.entries.filter((e) => e.seq < oldest)], more: shown.more };
}

/** Note who made `entries`, by id, in `authors`. */
export function noteAuthors(authors: Record<string, string>, entries: Entry[]): void {
	for (const entry of entries) {
		if (entry.user_id && entry.username && authors[entry.user_id] !== entry.username) authors[entry.user_id] = entry.username;
	}
}

/** The people to pick from: the project's members, and others who made edits here. */
export function people(
	members: { user_id: string; username: string }[],
	authors: Record<string, string>,
): { id: string; name: string }[] {
	const names = new Map(Object.entries(authors));
	for (const member of members) names.set(member.user_id, member.username);
	return [...names].map(([id, name]) => ({ id, name })).sort((a, b) => a.name.localeCompare(b.name));
}
