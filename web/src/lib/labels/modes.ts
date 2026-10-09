/**
 * Which voxels a brush stroke, a filled polygon, or an erase may change, and
 * the `only_if` the label writer is told (see `ml4paleo.labels.deltas`).
 *
 * Painting can go anywhere, or only where nothing is labeled yet, only where
 * something is (background counts), or only over the classes picked. Erasing
 * goes anywhere or only over the classes picked.
 */

/** The values an edit can name, which are the label values 1 to 254 (0 is unlabeled). */
export const LOWEST_CLASS = 1;
export const HIGHEST_CLASS = 254;

export type PaintMode = "any" | "unlabeled" | "labeled" | "classes";
export type EraseMode = "any" | "classes";

export interface ModeInfo<M extends string> {
	value: M;
	/** A few words for the control. */
	label: string;
	/** What it does, for a tooltip. */
	title: string;
}

export const PAINT_MODES: ModeInfo<PaintMode>[] = [
	{ value: "any", label: "Anywhere", title: "Paint over everything under the brush or inside the shape" },
	{ value: "unlabeled", label: "Only unlabeled", title: "Paint only where nothing is labeled yet" },
	{ value: "labeled", label: "Only labeled", title: "Paint only over voxels that already have a label, background too" },
	{ value: "classes", label: "Only classes", title: "Paint only over voxels of the classes you pick" },
];

export const ERASE_MODES: ModeInfo<EraseMode>[] = [
	{ value: "any", label: "Anything", title: "Erase every label under the brush" },
	{ value: "classes", label: "Only classes", title: "Erase only voxels of the classes you pick" },
];

export const PAINT_MODE_NAMES = PAINT_MODES.map((info) => info.value);
export const ERASE_MODE_NAMES = ERASE_MODES.map((info) => info.value);

/**
 * The `only_if` for a mode: "any", "unlabeled", or "labeled" as they are,
 * and for "classes" the values picked, once each and ascending, as the
 * server writes them ("class:2,3,5"). `classes` must hold at least one value
 * (`chosenClasses` always gives one).
 */
export function onlyIf(mode: PaintMode | EraseMode, classes: readonly number[]): string {
	if (mode !== "classes") return mode;
	return `class:${[...new Set(classes)].sort((a, b) => a - b).join(",")}`;
}

/**
 * The classes a "classes" mode goes by: the ones `chosen` that are among
 * `available` (the project's classes and background, which may not be the
 * ones chosen when the project changed or a class was removed), in the order
 * of `available`; and where none is, `fallback` if it's available, else
 * the first available. Never empty unless nothing is available.
 */
export function chosenClasses(chosen: readonly number[], available: readonly number[], fallback: number | null): number[] {
	const picked = available.filter((value) => chosen.includes(value));
	if (picked.length > 0) return picked;
	const one = fallback !== null && available.includes(fallback) ? fallback : available[0];
	return one === undefined ? [] : [one];
}

/**
 * `chosen` as a set of class values to remember: whole numbers from 1 to 254,
 * once each and ascending. A single number is a set of one (what was saved
 * before more than one class could be picked), and anything else is none.
 */
export function toClassSet(chosen: unknown): number[] {
	const listed = Array.isArray(chosen) ? chosen : typeof chosen === "number" ? [chosen] : [];
	const values = listed.filter((v): v is number => Number.isInteger(v) && v >= LOWEST_CLASS && v <= HIGHEST_CLASS);
	return [...new Set(values)].sort((a, b) => a - b);
}

/** `value` if it is one of `modes`, else `otherwise`. */
export function modeFrom<M extends string>(value: unknown, modes: readonly M[], otherwise: M): M {
	return typeof value === "string" && (modes as readonly string[]).includes(value) ? (value as M) : otherwise;
}

/**
 * The paint mode a browser saved: the mode itself, or from before there were
 * modes, `protectLabels` (a checkbox for painting only unlabeled voxels).
 */
export function savedPaintMode(saved: { paintMode?: unknown; protectLabels?: unknown }): PaintMode {
	if (saved.paintMode !== undefined) return modeFrom(saved.paintMode, PAINT_MODE_NAMES, "any");
	return saved.protectLabels === true ? "unlabeled" : "any";
}

/** Names as a sentence part: "bone", "bone and matrix", "bone, matrix, and tooth". */
const list = new Intl.ListFormat("en", { style: "long", type: "conjunction" });

/** A button's summary of the classes picked: "bone", or "bone + 2 more". */
export function summarize(names: readonly string[]): string {
	const [first] = names;
	if (first === undefined) return "No class";
	return names.length === 1 ? first : `${first} + ${names.length - 1} more`;
}

/**
 * Where a brush, a filled polygon, or an erase goes under a mode, for the
 * hint line: "anywhere" is left unsaid (empty), the rest as a phrase to follow
 * what the tool does ("paints" ...). `names` are the classes picked.
 */
export function describeWhere(mode: PaintMode | EraseMode, names: readonly string[]): string {
	switch (mode) {
		case "any":
			return "";
		case "unlabeled":
			return "only where nothing is labeled yet";
		case "labeled":
			return "only over labeled voxels, background too";
		case "classes":
			return names.length > 3 ? `only over ${names.length} classes` : `only over ${list.format(names)}`;
	}
}
