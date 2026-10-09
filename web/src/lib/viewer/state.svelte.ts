/**
 * What the views of one image share: the crosshair position, the zoom, the
 * display window, and the label overlay. The display preferences persist in
 * this browser.
 */

import {
	chosenClasses,
	ERASE_MODE_NAMES,
	type EraseMode,
	modeFrom,
	onlyIf,
	type PaintMode,
	savedPaintMode,
	toClassSet,
} from "../labels/modes";
import type { Polygon, PolygonMode } from "../labels/polygon";
import type { PlaneMask } from "../labels/raster";
import type { Plane, Vec3 } from "./tiles";

export type Layout = "four" | Plane["name"];
export const LAYOUTS: Layout[] = ["four", "xy", "xz", "yz"];

export type Tool = "navigate" | "brush" | "eraser" | "polygon" | "roi";

/** A finished brush or eraser stroke and the settings it was drawn with. */
export interface Stroke {
	plane: Plane;
	slice: number;
	mask: PlaneMask;
	erase: boolean;
	value: number;
	onlyIf: string;
	radius: number;
}

const PREFERENCES = "m4p.viewer";
const LABEL_OPACITY = 0.5;
// Labels fainter than this are as good as hidden.
const FAINTEST_LABELS = 0.1;

interface Preferences {
	opacity: number;
	layout: Layout;
	brushRadius: number;
	/** Where brushes and polygons paint, and for "classes" the class values. */
	paintMode: PaintMode;
	paintClasses: number[];
	/** Where the eraser erases, and for "classes" the class values. */
	eraseMode: EraseMode;
	eraseClasses: number[];
	roiDepth: number;
	showPrediction: boolean;
	predictionOpacity: number;
	showSegmentation: boolean;
	segmentationOpacity: number;
}

/**
 * What a browser may have saved: this version's preferences, and what earlier
 * versions saved in their place (`protectLabels`, a checkbox for painting
 * only unlabeled voxels; `paintClass` and `eraseClass`, one class).
 */
type Saved = Partial<Preferences> & { protectLabels?: unknown; paintClass?: unknown; eraseClass?: unknown };

function loadPreferences(): Saved {
	try {
		const saved: unknown = JSON.parse(localStorage.getItem(PREFERENCES) ?? "{}");
		return saved && typeof saved === "object" ? (saved as Saved) : {};
	} catch {
		return {};
	}
}

export class ViewerState {
	/** The crosshair, in level-0 voxels (z, y, x); every view centers on it. */
	position = $state<Vec3>([0, 0, 0]);
	/** Screen pixels per level-0 voxel along the finest axis. */
	zoom = $state(1);
	window = $state<[number, number]>([0, 1]);
	opacity = $state(LABEL_OPACITY);
	/** Labels show on every visit, even if they were hidden on the last. */
	showLabels = $state(true);
	showPrediction = $state(true);
	predictionOpacity = $state(0.35);
	showSegmentation = $state(true);
	segmentationOpacity = $state(0.45);
	layout = $state<Layout>("four");
	help = $state(false);
	/** Refit the image as views resize, until the user pans or zooms. */
	autoFit = true;
	tool = $state<Tool>("navigate");
	/** The class brushes and polygons paint. */
	activeClass = $state<number | null>(null);
	/** Brush radius in voxels of the finest axis. */
	brushRadius = $state(4);
	/**
	 * Where brushes and polygons paint (a polygon's Add; Subtract goes by its
	 * own rule): anywhere, only where nothing is labeled, only where something
	 * is, or only over the classes in `paintClasses` (label values; background
	 * is 1). Those may name classes this project lacks, as what's saved is
	 * kept across projects: `paintCondition` goes by the ones it has.
	 */
	paintMode = $state<PaintMode>("any");
	paintClasses = $state<number[]>([]);
	/** Where the eraser erases: anything, or only the classes in `eraseClasses`. */
	eraseMode = $state<EraseMode>("any");
	eraseClasses = $state<number[]>([]);
	/** The polygon being drawn; replaced, never changed in place. */
	polygon = $state.raw<Polygon | null>(null);
	/** Closing a polygon fills it with the active class, or cuts it out of that class. */
	polygonMode = $state<PolygonMode>("add");
	/** Alt and Shift as held now, which change what closing a polygon does. */
	held = $state.raw({ altKey: false, shiftKey: false });
	/** A polygon is being dragged out freehand, on its slice: the views hold still until it's let go. */
	lassoing = $state(false);
	/** Space is held: drag pans whatever the tool. */
	panning = $state(false);
	/** New ROIs: one-voxel slices, or cubes this many voxels deep. */
	roiDepth = $state(1);
	selectedRoi = $state<string | null>(null);

	constructor(
		public shape: Vec3,
		public aspect: Vec3,
	) {
		const saved = loadPreferences();
		if (typeof saved.opacity === "number" && saved.opacity >= FAINTEST_LABELS) this.opacity = Math.min(1, saved.opacity);
		if (saved.layout && LAYOUTS.includes(saved.layout)) this.layout = saved.layout;
		if (typeof saved.brushRadius === "number") this.brushRadius = Math.min(64, Math.max(0.5, saved.brushRadius));
		this.paintMode = savedPaintMode(saved);
		this.paintClasses = toClassSet(saved.paintClasses ?? saved.paintClass);
		this.eraseMode = modeFrom(saved.eraseMode, ERASE_MODE_NAMES, "any");
		this.eraseClasses = toClassSet(saved.eraseClasses ?? saved.eraseClass);
		if (typeof saved.roiDepth === "number") this.roiDepth = Math.min(512, Math.max(1, Math.round(saved.roiDepth)));
		if (typeof saved.showPrediction === "boolean") this.showPrediction = saved.showPrediction;
		if (typeof saved.predictionOpacity === "number") this.predictionOpacity = Math.min(1, Math.max(0, saved.predictionOpacity));
		if (typeof saved.showSegmentation === "boolean") this.showSegmentation = saved.showSegmentation;
		if (typeof saved.segmentationOpacity === "number") this.segmentationOpacity = Math.min(1, Math.max(0, saved.segmentationOpacity));
	}

	savePreferences(): void {
		try {
			const preferences: Preferences = {
				opacity: this.opacity,
				layout: this.layout,
				brushRadius: this.brushRadius,
				paintMode: this.paintMode,
				paintClasses: [...this.paintClasses],
				eraseMode: this.eraseMode,
				eraseClasses: [...this.eraseClasses],
				roiDepth: this.roiDepth,
				showPrediction: this.showPrediction,
				predictionOpacity: this.predictionOpacity,
				showSegmentation: this.showSegmentation,
				segmentationOpacity: this.segmentationOpacity,
			};
			localStorage.setItem(PREFERENCES, JSON.stringify(preferences));
		} catch {
			// Private windows and blocked storage just don't remember.
		}
	}

	/**
	 * The classes (label values, background 1 among them) the paint mode goes
	 * by when it is "classes", or the eraser's: those chosen that are among
	 * `available`, else the active class, else the first available.
	 */
	classesFor(tool: "paint" | "erase", available: readonly number[]): number[] {
		const chosen = tool === "paint" ? this.paintClasses : this.eraseClasses;
		return chosenClasses(chosen, available, this.activeClass);
	}

	/** The `only_if` for painting with a brush or filling a polygon, given the label values this project has. */
	paintCondition(available: readonly number[]): string {
		return onlyIf(this.paintMode, this.classesFor("paint", available));
	}

	/** The `only_if` for erasing, given the label values this project has. */
	eraseCondition(available: readonly number[]): string {
		return onlyIf(this.eraseMode, this.classesFor("erase", available));
	}

	/** Note which of Alt and Shift a key or pointer event says are held. */
	noteKeys(keys: { altKey: boolean; shiftKey: boolean }): void {
		if (keys.altKey !== this.held.altKey || keys.shiftKey !== this.held.shiftKey) {
			this.held = { altKey: keys.altKey, shiftKey: keys.shiftKey };
		}
	}

	/** Move the crosshair, keeping it inside the image. */
	moveTo(point: Vec3): void {
		this.position = point.map((p, axis) => Math.min(this.shape[axis]! - 0.5, Math.max(0, p))) as Vec3;
	}

	step(axis: number, by: number): void {
		const point = [...this.position] as Vec3;
		point[axis] = Math.floor(point[axis]!) + by + 0.5;
		this.moveTo(point);
	}

	/**
	 * Show the labels layer if it's hidden or too faint to see, as after an
	 * edit, which would otherwise seem to vanish once drawn.
	 */
	revealLabels(): void {
		this.showLabels = true;
		if (this.opacity < FAINTEST_LABELS) this.opacity = LABEL_OPACITY;
	}

	zoomBy(factor: number): void {
		this.zoom = Math.min(64, Math.max(1 / 512, this.zoom * factor));
	}

	nextLayout(): void {
		this.layout = LAYOUTS[(LAYOUTS.indexOf(this.layout) + 1) % LAYOUTS.length]!;
	}
}
