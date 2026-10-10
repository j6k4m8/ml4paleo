import { describe, expect, it } from "vitest";
import {
	action,
	ago,
	type Entry,
	everything,
	link,
	noteAuthors,
	origin,
	people,
	place,
	refreshed,
	toggles,
	what,
	who,
} from "./history";

function op(seq: number, extra: Partial<Entry> = {}): Entry {
	return {
		seq,
		kind: "edit",
		user_id: "u-ada",
		username: "ada",
		job_kind: null,
		source: 1,
		tool: { name: "brush", radius: 4 },
		bbox: [5, 20, 30, 6, 29, 42],
		live: true,
		accepted: null,
		target_seq: null,
		target: null,
		created_at: "2026-10-07T12:00:00Z",
		...extra,
	};
}

const accepted = (extra: Partial<NonNullable<Entry["accepted"]>> = {}) => ({
	kind: "proposal" as const,
	model_id: "m1",
	model_name: "rf one",
	v1_job_id: null,
	roi_id: "r1",
	...extra,
});

function undo(seq: number, target: Entry, kind: "undo" | "redo" = "undo"): Entry {
	return op(seq, { kind, tool: {}, source: target.source, bbox: target.bbox, target_seq: target.seq, target });
}

describe("what, who, and origin", () => {
	it("names people's tools", () => {
		expect(what(op(1))).toBe("Brush stroke");
		expect(what(op(1, { tool: { name: "eraser" } }))).toBe("Eraser stroke");
		expect(what(op(1, { tool: { name: "polygon", points: [] } }))).toBe("Polygon");
		expect(what(op(1, { tool: { name: "polygon-erase" } }))).toBe("Polygon erase");
		expect([who(op(1)), origin(op(1))]).toEqual(["ada", "By hand"]);
	});

	it("doesn't take an edit's word for what it was", () => {
		for (const name of ["accept-prediction", "decline-prediction", "v1-import", "constructor", "toString", 7, undefined]) {
			const edit = op(1, { tool: { name } });
			expect([action(edit), what(edit), origin(edit)]).toEqual(["edit", "Edit", "By hand"]);
		}
	});

	it("says which model accepted labels came from", () => {
		const edit = op(1, { source: 2, tool: { name: "accept-prediction" }, accepted: accepted() });
		expect([action(edit), what(edit), origin(edit)]).toEqual(["accept", "Accepted proposal", "From rf one"]);
		const v1 = op(2, { source: 2, accepted: accepted({ kind: "prediction", model_id: null, model_name: null, v1_job_id: "ab12cd" }) });
		expect([what(v1), origin(v1)]).toEqual(["Accepted prediction", "From v1 job ab12cd"]);
		expect(origin(op(3, { source: 2, accepted: accepted({ model_name: null }) }))).toBe("From a model");
	});

	it("names checked declines and explicit restores", () => {
		const declined = op(1, { source: 6, tool: { name: "decline-prediction" }, accepted: accepted() });
		expect([action(declined), what(declined), origin(declined)]).toEqual(["decline", "Declined proposal", "From rf one"]);
		const restored = op(2, { tool: { name: "restore-declined" } });
		expect([action(restored), what(restored), origin(restored)]).toEqual(["restore-declined", "Restored suggestion", "By hand"]);
	});

	it("names imports and other sources", () => {
		const imported = op(1, { user_id: null, username: null, job_kind: "v1.labels", tool: { name: "v1-import" } });
		expect([action(imported), who(imported), what(imported), origin(imported)]).toEqual([
			"import",
			"v1 import",
			"v1 annotation",
			"By hand",
		]);
		// A file someone imported: theirs, and named.
		const file = op(2, { source: 5, job_kind: "labels.import", tool: { name: "label-import", file: "bone.tif" } });
		expect([action(file), who(file), what(file), origin(file)]).toEqual(["import", "ada", "From bone.tif", "Imported"]);
		expect(what(op(2, { job_kind: "labels.import", tool: {} }))).toBe("Labels from a file");
		expect(who(op(1, { username: null, job_kind: "propagate" }))).toBe("A job");
		expect(who(op(1, { user_id: null, username: null }))).toBe("Someone");
		expect(origin(op(1, { source: 4 }))).toBe("Propagated");
		expect(origin(op(1, { source: 5 }))).toBe("Imported");
		expect(origin(op(1, { source: 9 }))).toBe("Unknown");
	});

	it("describes undos and redos by their edit", () => {
		const edit = op(3, { source: 2, accepted: accepted(), live: false });
		const undone = undo(7, edit);
		expect([action(undone), what(undone), origin(undone)]).toEqual(["undo", "Undid #3", "From rf one"]);
		expect(what(undo(9, edit, "redo"))).toBe("Redid #3");
	});
});

describe("place", () => {
	it("gives the size and corner in x, y, z", () => {
		expect(place([5, 20, 30, 6, 29, 42])).toEqual({ size: "12 × 9 × 1", x: 30, y: 20, z: 5 });
	});
});

describe("ago", () => {
	const now = Date.parse("2026-10-07T12:00:00Z");

	it("counts back minutes and hours, then gives the date", () => {
		expect(ago("2026-10-07T11:59:30Z", now)).toBe("just now");
		expect(ago("2026-10-07T11:55:00Z", now)).toBe("5 min ago");
		expect(ago("2026-10-07T09:10:00Z", now)).toBe("2 h ago");
		expect(ago("2026-10-05T12:00:00Z", now)).toBe(new Date("2026-10-05T12:00:00Z").toLocaleString());
	});
});

describe("link", () => {
	it("opens the annotator on the box, and on the ROI labels were accepted in", () => {
		expect(link("p1", op(1))).toBe("/p/p1/annotate?box=5,20,30,6,29,42");
		const edit = op(2, { source: 2, accepted: accepted() });
		expect(link("p1", edit)).toBe("/p/p1/annotate?roi=r1&box=5,20,30,6,29,42");
		expect(link("p1", undo(3, edit))).toBe("/p/p1/annotate?roi=r1&box=5,20,30,6,29,42");
	});

	it("opens the annotator on the box alone for labels accepted in a view, which have no ROI", () => {
		const edit = op(2, { source: 2, bbox: [3, 0, 0, 4, 80, 100], tool: { name: "accept-prediction", box: [3, 0, 0, 4, 80, 100] }, accepted: accepted({ kind: "prediction", roi_id: null }) });
		expect([action(edit), what(edit), origin(edit)]).toEqual(["accept", "Accepted prediction", "From rf one"]);
		expect(link("p1", edit)).toBe("/p/p1/annotate?box=3,0,0,4,80,100");
		expect(link("p1", undo(3, edit))).toBe("/p/p1/annotate?box=3,0,0,4,80,100");
	});
});

describe("toggles", () => {
	it("undoes live edits and redoes undone ones", () => {
		const buttons = toggles([op(2, { live: false }), op(1)]);
		expect(buttons.get(2)).toEqual({ seq: 2, action: "redo" });
		expect(buttons.get(1)).toEqual({ seq: 1, action: "undo" });
	});

	it("reverses only the newest undo or redo of an edit, while it holds", () => {
		const edit = op(1, { live: false });
		// Undone, redone, and undone again.
		const buttons = toggles([undo(4, edit), undo(3, edit, "redo"), undo(2, edit), edit]);
		expect(buttons.get(4)).toEqual({ seq: 1, action: "redo" });
		expect(buttons.has(3) || buttons.has(2)).toBe(false);
		expect(buttons.get(1)).toEqual({ seq: 1, action: "redo" });
		// Redone by someone not listed: the undo no longer holds.
		const redone = toggles([undo(2, { ...edit, live: true })]);
		expect(redone.size).toBe(0);
		expect(toggles([undo(2, { ...edit, live: true }, "redo")]).get(2)).toEqual({ seq: 1, action: "undo" });
	});
});

describe("everything", () => {
	it("follows the pages back until one comes up short", async () => {
		const all = [9, 8, 7, 6, 5].map((seq) => op(seq));
		const asked: (number | null)[] = [];
		const page = async (before: number | null) => {
			asked.push(before);
			return all.filter((e) => before === null || e.seq < before).slice(0, 2);
		};
		expect((await everything(page, 2)).map((e) => e.seq)).toEqual([9, 8, 7, 6, 5]);
		expect(asked).toEqual([null, 8, 6]);
		expect(await everything(async () => [], 2)).toEqual([]);
	});
});

describe("refreshed", () => {
	const seqs = (listing: { entries: Entry[] }) => listing.entries.map((e) => e.seq);

	it("starts with the newest page", () => {
		expect(refreshed([op(9), op(8)], undefined, { entries: [], more: false }, 2)).toEqual({ entries: [op(9), op(8)], more: true });
		expect(refreshed([op(9)], undefined, { entries: [], more: true }, 2)).toEqual({ entries: [op(9)], more: false });
	});

	it("takes every entry back to the oldest shown, as it is now", () => {
		const shown = { entries: [op(10), op(8), op(6, { live: true })], more: true };
		const fresh = [op(12), op(11), op(10), op(8), op(6, { live: false })];
		const after = refreshed(fresh, 6, shown, 2);
		expect(seqs(after)).toEqual([12, 11, 10, 8, 6]);
		expect(after.entries.at(-1)?.live).toBe(false);
		expect(after.more).toBe(true);
	});

	it("keeps older entries loaded while it reloaded", () => {
		const shown = { entries: [op(10), op(8), op(6), op(4), op(2)], more: false };
		expect(seqs(refreshed([op(12), op(10), op(8), op(6)], 6, shown, 2))).toEqual([12, 10, 8, 6, 4, 2]);
		expect(refreshed([op(12), op(10), op(8), op(6)], 6, shown, 2).more).toBe(false);
	});
});

describe("people", () => {
	it("lists members and anyone else who made an edit, by name", () => {
		const members = [
			{ user_id: "u-bob", username: "bob" },
			{ user_id: "u-ada", username: "ada" },
		];
		const authors: Record<string, string> = {};
		const left = op(1, { user_id: "u-cy", username: "cy" });
		const job = op(2, { user_id: null, username: null, job_kind: "v1.labels" });
		noteAuthors(authors, [op(3), left, job]);
		expect(authors).toEqual({ "u-ada": "ada", "u-cy": "cy" });
		expect(people(members, authors)).toEqual([
			{ id: "u-ada", name: "ada" },
			{ id: "u-bob", name: "bob" },
			{ id: "u-cy", name: "cy" },
		]);
	});
});
