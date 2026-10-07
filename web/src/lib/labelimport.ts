/**
 * Labels from a file: a TIFF stack, or a zip of TIFF or PNG slices, the
 * size of the project's image. The server checks the file and counts the
 * values in it; the person says what each value becomes (background, a
 * class, a new class, or nothing); then a worker brings the labels in.
 */

import type { Pipeline } from "./types";
import type { LabelClass } from "./viewer/labels";

export const BACKGROUND = 1;

export interface LabelImport {
	id: string;
	filename: string;
	upload_id: string;
	created_at: string;
	created_by: string | null;
	state: "checking" | "ready" | "importing" | "done" | "failed" | "cancelled" | "expired";
	error: string | null;
	/** The check's pipeline, then the import's once it starts. */
	pipeline: Pipeline;
	/** Once checked: the labels' (z, y, x) size and each value's voxels. */
	shape_zyx: [number, number, number] | null;
	values: { value: number; voxels: number }[] | null;
	/** Once started: what each value became, and whether it replaced labels. */
	lookup: [number, number][] | null;
	overwrite: boolean | null;
}

/** What a value in the file becomes. */
export type Target =
	| { kind: "skip" }
	| { kind: "label"; label: number }
	| { kind: "new"; name: string; color: string };

/** Colors for new classes, taken in turn, skipping ones in use. */
export const CLASS_COLORS = [
	"#e5484d",
	"#46a758",
	"#3e63dd",
	"#f2c14e",
	"#d6409f",
	"#12a594",
	"#f76b15",
	"#8e4ec6",
	"#0090ff",
	"#a18072",
];

/** The first class color not in `used`, or one in turn once all are. */
export function nextColor(used: string[]): string {
	const taken = new Set(used.map((c) => c.toLowerCase()));
	return CLASS_COLORS.find((c) => !taken.has(c)) ?? CLASS_COLORS[used.length % CLASS_COLORS.length]!;
}

/** Every nonzero value as a new class named after it. */
export function newClasses(values: number[], classes: LabelClass[]): Map<number, Target> {
	const used = classes.map((c) => c.color);
	const targets = new Map<number, Target>();
	for (const value of values) {
		if (value === 0) {
			targets.set(value, { kind: "skip" });
			continue;
		}
		const color = nextColor(used);
		used.push(color);
		targets.set(value, { kind: "new", name: String(value), color });
	}
	return targets;
}

/**
 * First choices: 0 stays unlabeled; a value a class is named after becomes
 * that class; the only other value becomes the only class; the rest become
 * new classes named after them.
 */
export function defaultTargets(values: number[], classes: LabelClass[]): Map<number, Target> {
	const targets = newClasses(values, classes);
	const others = values.filter((v) => v !== 0);
	for (const value of others) {
		const named = classes.find((c) => c.name === String(value));
		if (named) targets.set(value, { kind: "label", label: named.value });
	}
	const only = classes[0];
	if (others.length === 1 && classes.length === 1 && only) targets.set(others[0]!, { kind: "label", label: only.value });
	return targets;
}

export interface StartBody {
	mapping: ({ value: number; label: number } | { value: number; new_class: { name: string; color: string } })[];
	overwrite: boolean;
}

/** The request that starts an import, or why it can't start yet. */
export function startBody(targets: Map<number, Target>, overwrite: boolean): StartBody | string {
	const mapping: StartBody["mapping"] = [];
	for (const [value, target] of targets) {
		if (target.kind === "label") mapping.push({ value, label: target.label });
		else if (target.kind === "new") {
			const name = target.name.trim();
			if (!name) return "Give each new class a name.";
			mapping.push({ value, new_class: { name, color: target.color } });
		}
	}
	if (mapping.length === 0) return "Choose what at least one value becomes.";
	return { mapping, overwrite };
}

/** Whether a file's name looks like a label file the server reads. */
export function labelFile(name: string): boolean {
	return /\.(tiff?|png|zip)$/i.test(name);
}
