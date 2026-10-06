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
