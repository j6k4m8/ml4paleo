/**
 * The viewer's keys, in one table that drives both the handlers and the
 * help overlay (`?`).
 */

export type Action =
	| "slice-next"
	| "slice-previous"
	| "zoom-in"
	| "zoom-out"
	| "fit"
	| "layout"
	| "labels"
	| "help";

export interface Binding {
	action: Action;
	keys: string[];
	label: string;
}

export const KEYMAP: Binding[] = [
	{ action: "slice-next", keys: [".", "ArrowUp"], label: "Next slice (with Shift: 10)" },
	{ action: "slice-previous", keys: [",", "ArrowDown"], label: "Previous slice (with Shift: 10)" },
	{ action: "zoom-in", keys: ["=", "+"], label: "Zoom in" },
	{ action: "zoom-out", keys: ["-", "_"], label: "Zoom out" },
	{ action: "fit", keys: ["0"], label: "Fit the image" },
	{ action: "layout", keys: ["l"], label: "Next layout (four views, XY, XZ, YZ)" },
	{ action: "labels", keys: ["v"], label: "Show or hide labels" },
	{ action: "help", keys: ["?"], label: "Show or hide these keys" },
];

/** What the mouse does, for the help overlay. */
export const MOUSE: [string, string][] = [
	["Wheel", "Next or previous slice"],
	["Ctrl + wheel, or pinch", "Zoom"],
	["Drag", "Pan"],
	["Click", "Move the crosshair there"],
];

const BY_KEY = new Map(KEYMAP.flatMap((binding) => binding.keys.map((key) => [key, binding.action] as const)));

/** The action for a key press, ignoring presses meant for form fields. */
export function actionFor(event: KeyboardEvent): Action | undefined {
	if (event.ctrlKey || event.metaKey || event.altKey) return undefined;
	const target = event.target as HTMLElement | null;
	if (target && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))) {
		return undefined;
	}
	return BY_KEY.get(event.key.length === 1 ? event.key.toLowerCase() : event.key) ?? BY_KEY.get(event.key);
}
