/**
 * Background: the class every project has (label value 1), painted to say
 * what the model should leave alone. It isn't a row in the project's
 * classes, so it is added to the lists the person picks from.
 */

import type { LabelClass } from "./labels";

export const BACKGROUND_VALUE = 1;

export const BACKGROUND_CLASS: LabelClass = { value: BACKGROUND_VALUE, name: "Background", color: "#7c8aa5" };

/** How opaque painted background is where labels draw (of 255): a haze, so the image still shows. */
export const BACKGROUND_ALPHA = 170;

/** The project's classes with background first, as the person picks from them. */
export function withBackground(classes: LabelClass[]): LabelClass[] {
	return [BACKGROUND_CLASS, ...classes];
}
