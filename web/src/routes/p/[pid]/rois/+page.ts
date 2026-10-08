import { type Load, redirect } from "@sveltejs/kit";
import { SHOW_ROIS } from "#lib/features.ts";

// While ROIs are hidden, an old link to this page goes to the Labels page.
export const load: Load = ({ params }) => {
	if (!SHOW_ROIS) redirect(307, `/p/${params.pid}/labels`);
};
