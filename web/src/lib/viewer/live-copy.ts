import type { ModelPlugin } from "./live";
import type { Plane } from "./tiles";

/** Name the exact 2D target even when three slice panes are visible. */
export function reviewSlice(plane: Plane, slice: number): string {
	return `${plane.name.toUpperCase()} slice (${"ZYX"[plane.normal]} ${slice})`;
}

/** Explain the selected method's timing without implying every method trains automatically. */
export function learningSchedule(mode: ModelPlugin["capabilities"]["learning"] | undefined): string {
	if (mode === "debounced") return "Learns from new labels after you pause painting.";
	if (mode === "periodic") return "Learns from new labels at intervals while you keep painting.";
	if (mode === "manual") return "This method does not learn automatically. Use the Models page to teach it from your labels.";
	return "Loading the available suggestion methods…";
}

/** Give common failures a useful next action; raw server details stay in the disclosure. */
export function learningHelp(detail: string): string {
	if (/quota|as many trained models|model.*limit/i.test(detail)) {
		return "No room to save what the app learns next. Ask the project owner to raise the model limit, or free space on Models, then try again.";
	}
	if (/add a label class|label something|no labeled voxels|at least two classes/i.test(detail)) {
		return "Paint a few examples of what you want to find and some Background, then try again.";
	}
	return "Try again. If it still fails, open the error details below.";
}
