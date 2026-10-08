/**
 * Which chunks of the layers drawn over the image a view draws and loads.
 *
 * The labels come at the levels of the image's pyramid, so a view picks the
 * level that fits it as the image does and loads the coarser ones too, coarsest
 * first, to show while the finer ones arrive. A model's prediction and the
 * segmentation have full resolution only: past a limit they're left out.
 */

import { labelId, labelKey } from "./labels";
import type { LabelTile } from "./plane";
import {
	countDrawn,
	countTiles,
	type Level,
	sliceIndex,
	TILE_HYSTERESIS,
	type TileKey,
	tilesShown,
	tilesToLoad,
	type View,
	viewLevel,
} from "./tiles";

/**
 * Most full resolution chunks a view draws of a layer that has no coarser
 * levels (each is 256 KiB, and the three views share a cache of 128 MiB). More
 * than this in view and the layer is left out, until the view needs this many
 * times fewer (TILE_HYSTERESIS), so zooming near the limit doesn't flicker it.
 */
export const MAX_OVERLAY_TILES = 128;

/**
 * Most chunks a view draws of the labels when they have coarser levels: the
 * level it shows and the coarser ones under it, past which it shows a coarser
 * level (each chunk is 256 KiB, and the three views share a cache of 256 MiB).
 */
export const MAX_LABEL_TILES = 256;

/** Whether `full`'s chunks in view are too many to draw (`hidden`: they were too many before). */
export function overlayHidden(full: Level, view: View, hidden: boolean): boolean {
	return countTiles(full, view) > (hidden ? MAX_OVERLAY_TILES / TILE_HYSTERESIS : MAX_OVERLAY_TILES);
}

export interface LabelsView {
	/** The level the view shows the labels at. */
	level: Level;
	/**
	 * Whether even that is too much to draw (the coarsest level's chunks in
	 * view are too many, or without coarser levels, the full resolution
	 * ones), so only this page's latest edits show.
	 */
	hidden: boolean;
	/** The chunks to load, by id, most wanted first. */
	wanted: string[];
	/** The first of `wanted` the view is drawing (the rest load ahead). */
	shown: string[];
	/** The chunks to draw, finest first: those at hand are all the view has to go on. */
	draw: LabelTile[];
}

/**
 * What a view shows of the labels `levels`. It shows the level that fits the
 * view (`current` is the level it showed last, which it keeps a little past
 * the point it'd leave it, as the image does), with the coarser levels under
 * it, which load first. Wherever there are finer chunks at hand (`cached`:
 * ids of chunks loaded, such as the finer level the view has just left), they
 * show too, and this page's latest edits (`recent`, ids of full resolution
 * chunks) always do, at full resolution, so an edit doesn't wait for the
 * levels over it to catch up with it. Each point shows the finest of these.
 */
export function labelsView(
	levels: Level[],
	view: View,
	{ current, hidden, recent, cached }: { current?: number; hidden: boolean; recent: readonly string[]; cached: Iterable<string> },
): LabelsView {
	const limit = levels.length > 1 ? MAX_LABEL_TILES : MAX_OVERLAY_TILES;
	const level = viewLevel(levels, view, current, limit);
	const tooMany = countDrawn(levels, level, view) > (hidden ? limit / TILE_HYSTERESIS : limit);
	const full = levels[0]!;
	const edited = tilesShown(recent.map(labelKey), full, view).slice(0, MAX_OVERLAY_TILES);
	const wanted = new Map<string, TileKey>();
	const shown = new Set<string>();
	const drawn = new Map<string, TileKey>();
	const want = (key: TileKey, draws: boolean) => {
		const id = labelId(key);
		wanted.set(id, key);
		if (draws) {
			shown.add(id);
			drawn.set(id, key);
		}
	};
	for (const key of edited) want(key, true);
	if (!tooMany) {
		const { tiles, shown: count } = tilesToLoad(levels, level, view);
		tiles.forEach((key, place) => want(key, place < count));
		// What the view has of the levels finer than this one, which it shows until this one's arrive.
		const finer = new Map<number, TileKey[]>();
		for (const id of cached) {
			const key = labelKey(id);
			if (key.level < level.index && levels[key.level]) finer.set(key.level, [...(finer.get(key.level) ?? []), key]);
		}
		for (const [index, keys] of finer) {
			for (const key of tilesShown(keys, levels[index]!, view)) drawn.set(labelId(key), key);
		}
	}
	const draw = [...drawn.entries()]
		.map(([id, key]) => {
			const of = levels[key.level]!;
			return { id, key, slice: sliceIndex(of, view), scale: of.scale };
		})
		.sort((a, b) => a.key.level - b.key.level);
	return { level, hidden: tooMany, wanted: [...wanted.keys()], shown: [...shown], draw };
}
