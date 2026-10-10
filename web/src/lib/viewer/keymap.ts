/**
 * The viewer's keys, in one table that drives both the handlers and the
 * help overlay (`?`). `mod` is Ctrl, or ⌘ on a Mac.
 */

import { SHOW_ROIS } from "../features";

export type Action =
	| "slice-next"
	| "slice-previous"
	| "zoom-in"
	| "zoom-out"
	| "fit"
	| "layout"
	| "labels"
	| "prediction"
	| "navigate"
	| "brush"
	| "eraser"
	| "polygon"
	| "roi"
	| "next-roi"
	| "complete-roi"
	| "accept"
	| "accept-tool"
	| "decline"
	| "smaller"
	| "bigger"
	| "class"
	| "close-polygon"
	| "remove-point"
	| "cancel"
	| "undo"
	| "redo"
	| "help";

export interface Binding {
	action: Action;
	keys: string[];
	label: string;
}

const ROI_ACTIONS = new Set<Action>(["roi", "next-roi", "complete-roi"]);

const ALL_KEYS: Binding[] = [
	// With Shift, "." and "," arrive as ">" and "<" (on US keyboards).
	{ action: "slice-next", keys: [".", ">", "ArrowUp"], label: "Next slice (with Shift: 10)" },
	{ action: "slice-previous", keys: [",", "<", "ArrowDown"], label: "Previous slice (with Shift: 10)" },
	{ action: "zoom-in", keys: ["=", "+"], label: "Zoom in" },
	{ action: "zoom-out", keys: ["-", "_"], label: "Zoom out" },
	{ action: "fit", keys: ["0"], label: "Fit the image" },
	{ action: "layout", keys: ["l"], label: "Next layout (four views, XY, XZ, YZ)" },
	{ action: "labels", keys: ["v"], label: "Show or hide labels" },
	{ action: "prediction", keys: ["m"], label: "Show or hide the model's prediction" },
	{ action: "navigate", keys: ["n"], label: "Navigate" },
	{ action: "brush", keys: ["b"], label: "Brush" },
	{ action: "eraser", keys: ["e"], label: "Eraser" },
	{ action: "polygon", keys: ["p"], label: "Polygon: click points or drag freehand; Add fills, Subtract cuts out" },
	{ action: "accept-tool", keys: ["i"], label: "Accept tool: brush or polygon over visible foreground predictions" },
	{ action: "roi", keys: ["r"], label: "Draw an ROI" },
	{ action: "next-roi", keys: ["g"], label: "Go to the next open ROI" },
	{ action: "complete-roi", keys: ["c"], label: "Mark the selected ROI complete (with Shift: reopen)" },
	{
		action: "accept",
		keys: ["a"],
		label: SHOW_ROIS
			? "Accept the prediction in the selected ROI, or in this view if none is selected, as labels (fills unlabeled voxels)"
			: "Accept the prediction in this view as labels (fills unlabeled voxels)",
	},
	{
		action: "decline",
		keys: ["x"],
		label: SHOW_ROIS
			? "Decline foreground suggestions in the selected ROI, or in this view if none is selected"
			: "Decline foreground suggestions in this view",
	},
	{ action: "smaller", keys: ["["], label: "Smaller brush" },
	{ action: "bigger", keys: ["]"], label: "Bigger brush" },
	{ action: "class", keys: ["1", "2", "3", "4", "5", "6", "7", "8", "9"], label: "Choose a class (1 is Background)" },
	{ action: "close-polygon", keys: ["Enter"], label: "Close the polygon; Accept keeps predicted classes (paint polygon: Alt cuts, Shift fills)" },
	{ action: "remove-point", keys: ["Backspace"], label: "Remove the polygon's last point" },
	{ action: "cancel", keys: ["Escape"], label: "Drop the polygon, then go back to navigating" },
	{ action: "undo", keys: ["mod+z"], label: "Undo your last edit" },
	{ action: "redo", keys: ["mod+shift+z", "mod+y"], label: "Redo" },
	{ action: "help", keys: ["?"], label: "Show or hide these keys (Esc closes)" },
];

/** The keys the viewer answers: ROIs' only while ROIs are shown (accepting works in a view without them). */
export const KEYMAP: Binding[] = ALL_KEYS.filter((binding) => SHOW_ROIS || !ROI_ACTIONS.has(binding.action));

/** What the mouse does, for the help overlay. */
const ALL_MOUSE: [string, string][] = [
	["Wheel", "Next or previous slice"],
	["Ctrl + wheel, or pinch", "Zoom"],
	["Drag (navigating), middle drag, or Space + drag", "Pan"],
	["Right-click (Mac: or Ctrl + click), with any tool", "Move the crosshair there"],
	["Tap (touch or pen, navigating)", "Move the crosshair there"],
	["Drag (brush, eraser)", "Paint"],
	["Drag (Accept brush)", "Accept visible foreground predictions; keep their classes"],
	["Click or drag (Accept polygon)", "Outline predictions to accept; close with Enter or the first point"],
	["Click (polygon)", "Add a point; on the first point, close the polygon"],
	["Drag (polygon)", "Draw freehand; letting go closes the polygon"],
	["Double-click (polygon)", "Close the polygon"],
	["Alt or Shift while closing (polygon)", "Cut out of the active class, or fill, whatever the mode"],
	["Drag (ROI)", "Draw an ROI on this slice"],
];

export const MOUSE = ALL_MOUSE.filter(([what]) => SHOW_ROIS || !what.includes("ROI"));

const MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

/**
 * Whether a press is a right-click: the right button, or on a Mac Ctrl with
 * the left (where the system makes it one). In a view it moves the crosshair
 * to where it points, whatever the tool; a plain left click never does, so a
 * stray click can't move you. Elsewhere Ctrl is the zoom key, held down while
 * painting, so it doesn't count.
 */
export function isRightClick(event: Pick<PointerEvent, "button" | "ctrlKey">, mac = MAC): boolean {
	return event.button === 2 || (mac && event.button === 0 && event.ctrlKey);
}

const BY_KEY = new Map(KEYMAP.flatMap((binding) => binding.keys.map((key) => [key, binding.action] as const)));

/** The key press as the keymap spells it. */
export function comboOf(event: KeyboardEvent): string {
	const name = event.key.length === 1 ? event.key.toLowerCase() : event.key;
	if (!(event.ctrlKey || event.metaKey)) return name;
	// Shortcuts follow the key's position, so Ctrl+Z works on any layout.
	const letter = /^Key([A-Z])$/.exec(event.code ?? "")?.[1]?.toLowerCase();
	return `mod+${event.shiftKey ? "shift+" : ""}${letter ?? name}`;
}

const NON_TEXT_INPUTS = new Set(["checkbox", "radio", "range", "button", "submit", "reset", "color", "file"]);

/**
 * Whether a key press belongs to the focused element rather than the
 * viewer: typing in a field, Space or Enter on a button or checkbox, arrows
 * on a slider, radio button, or button.
 */
export function forFocused(event: KeyboardEvent): boolean {
	const target = event.target as (HTMLElement & { type?: string }) | null;
	if (!target?.tagName) return false;
	const tag = target.tagName;
	if (target.isContentEditable || tag === "TEXTAREA" || tag === "SELECT") return true;
	if (tag === "INPUT" && !NON_TEXT_INPUTS.has(target.type ?? "text")) return true;
	const control = tag === "BUTTON" || tag === "A" || tag === "INPUT" || tag === "SUMMARY";
	if (control && (event.key === " " || event.key === "Enter")) return true;
	if (!event.key.startsWith("Arrow")) return false;
	return tag === "BUTTON" || (tag === "INPUT" && (target.type === "range" || target.type === "radio"));
}

// Alt changes how a polygon closes, so it may be held while drawing one;
// with it, only these keys are the viewer's.
const WITH_ALT = new Set(["Enter", "Backspace", "Escape"]);

/** The action for a key press, or undefined if it isn't the viewer's. */
export function actionFor(event: KeyboardEvent): Action | undefined {
	if (event.altKey && !WITH_ALT.has(event.key)) return undefined;
	if (forFocused(event)) return undefined;
	return BY_KEY.get(comboOf(event));
}
