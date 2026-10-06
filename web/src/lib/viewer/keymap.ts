/**
 * The viewer's keys, in one table that drives both the handlers and the
 * help overlay (`?`). `mod` is Ctrl, or ⌘ on a Mac.
 */

export type Action =
	| "slice-next"
	| "slice-previous"
	| "zoom-in"
	| "zoom-out"
	| "fit"
	| "layout"
	| "labels"
	| "navigate"
	| "brush"
	| "eraser"
	| "polygon"
	| "roi"
	| "next-roi"
	| "complete-roi"
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

export const KEYMAP: Binding[] = [
	// With Shift, "." and "," arrive as ">" and "<" (on US keyboards).
	{ action: "slice-next", keys: [".", ">", "ArrowUp"], label: "Next slice (with Shift: 10)" },
	{ action: "slice-previous", keys: [",", "<", "ArrowDown"], label: "Previous slice (with Shift: 10)" },
	{ action: "zoom-in", keys: ["=", "+"], label: "Zoom in" },
	{ action: "zoom-out", keys: ["-", "_"], label: "Zoom out" },
	{ action: "fit", keys: ["0"], label: "Fit the image" },
	{ action: "layout", keys: ["l"], label: "Next layout (four views, XY, XZ, YZ)" },
	{ action: "labels", keys: ["v"], label: "Show or hide labels" },
	{ action: "navigate", keys: ["n"], label: "Navigate" },
	{ action: "brush", keys: ["b"], label: "Brush" },
	{ action: "eraser", keys: ["e"], label: "Eraser" },
	{ action: "polygon", keys: ["p"], label: "Polygon (with Alt when closing: erase that class)" },
	{ action: "roi", keys: ["r"], label: "Draw an ROI" },
	{ action: "next-roi", keys: ["g"], label: "Go to the next open ROI" },
	{ action: "complete-roi", keys: ["c"], label: "Mark the selected ROI complete (with Shift: reopen)" },
	{ action: "smaller", keys: ["["], label: "Smaller brush" },
	{ action: "bigger", keys: ["]"], label: "Bigger brush" },
	{ action: "class", keys: ["1", "2", "3", "4", "5", "6", "7", "8", "9"], label: "Choose a class" },
	{ action: "close-polygon", keys: ["Enter"], label: "Close the polygon (or double-click)" },
	{ action: "remove-point", keys: ["Backspace"], label: "Remove the polygon's last point" },
	{ action: "cancel", keys: ["Escape"], label: "Drop the polygon, then go back to navigating" },
	{ action: "undo", keys: ["mod+z"], label: "Undo your last edit" },
	{ action: "redo", keys: ["mod+shift+z", "mod+y"], label: "Redo" },
	{ action: "help", keys: ["?"], label: "Show or hide these keys (Esc closes)" },
];

/** What the mouse does, for the help overlay. */
export const MOUSE: [string, string][] = [
	["Wheel", "Next or previous slice"],
	["Ctrl + wheel, or pinch", "Zoom"],
	["Drag (navigating), middle drag, or Space + drag", "Pan"],
	["Click (navigating)", "Move the crosshair there"],
	["Drag (brush, eraser)", "Paint"],
	["Click (polygon)", "Add a point"],
	["Drag (ROI)", "Draw an ROI on this slice"],
];

const BY_KEY = new Map(KEYMAP.flatMap((binding) => binding.keys.map((key) => [key, binding.action] as const)));

/** The key press as the keymap spells it. */
export function comboOf(event: KeyboardEvent): string {
	const name = event.key.length === 1 ? event.key.toLowerCase() : event.key;
	if (!(event.ctrlKey || event.metaKey)) return name;
	return `mod+${event.shiftKey ? "shift+" : ""}${name}`;
}

/** The action for a key press, ignoring presses meant for form fields. */
export function actionFor(event: KeyboardEvent): Action | undefined {
	if (event.altKey && event.key !== "Enter") return undefined;
	const target = event.target as HTMLElement | null;
	if (target && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) {
		return undefined;
	}
	return BY_KEY.get(comboOf(event));
}
