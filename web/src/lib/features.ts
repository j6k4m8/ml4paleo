/**
 * Parts of the app that exist but aren't shown yet.
 *
 * ROIs (boxes that mark where labeling is done, or held out to score models)
 * are hidden for now: labeling is simpler when the model needs just two
 * kinds of labels, what you're looking for and background. The server keeps
 * them and training still reads any that exist. Turn this on to bring back
 * their page, tool, keys, and buttons. (Accepting the prediction in a view
 * needs no ROIs, so it isn't hidden.)
 */
export const SHOW_ROIS = false;
