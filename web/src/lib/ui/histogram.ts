/**
 * Which end of the display window a press at `value` grabs: the end on the
 * side pressed when it's outside them (so ends that meet can still be pulled
 * apart), or else the nearer one, black on a tie.
 */
export function handleAt(value: number, [black, white]: [number, number]): 0 | 1 {
	if (value > white) return 1;
	if (value < black) return 0;
	return value - black <= white - value ? 0 : 1;
}

/** Round the controls, not the underlying display window: float images still need fractional contrast. */
export function displayLevel(value: number): number {
	return Math.round(value);
}

/** A number-field edit commits a whole level; clearing it must not put NaN into the renderer. */
export function editLevel(window: [number, number], end: 0 | 1, value: number | undefined): [number, number] {
	if (typeof value !== "number" || !Number.isFinite(value)) return window;
	const next: [number, number] = [...window];
	next[end] = Math.round(value);
	return next;
}
