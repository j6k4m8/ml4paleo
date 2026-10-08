/**
 * What the views of one image share: the crosshair position, the zoom, the
 * display window, and the label overlay. The display preferences persist in
 * this browser.
 */

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

interface Preferences {
	opacity: number;
	showLabels: boolean;
	layout: Layout;
	brushRadius: number;
	protectLabels: boolean;
	roiDepth: number;
	showPrediction: boolean;
	predictionOpacity: number;
	showSegmentation: boolean;
	segmentationOpacity: number;
}

function loadPreferences(): Partial<Preferences> {
	try {
		const saved: unknown = JSON.parse(localStorage.getItem(PREFERENCES) ?? "{}");
		return saved && typeof saved === "object" ? (saved as Partial<Preferences>) : {};
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
	opacity = $state(0.5);
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
	/** Paint only voxels without a label. */
	protectLabels = $state(false);
	/** The polygon being drawn; replaced, never changed in place. */
	polygon = $state.raw<Polygon | null>(null);
	/** Closing a polygon fills it with the active class, or cuts it out of that class. */
	polygonMode = $state<PolygonMode>("add");
	/** Alt and Shift as held now, which change what closing a polygon does. */
	held = $state.raw({ altKey: false, shiftKey: false });
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
		if (typeof saved.opacity === "number") this.opacity = Math.min(1, Math.max(0, saved.opacity));
		if (typeof saved.showLabels === "boolean") this.showLabels = saved.showLabels;
		if (saved.layout && LAYOUTS.includes(saved.layout)) this.layout = saved.layout;
		if (typeof saved.brushRadius === "number") this.brushRadius = Math.min(64, Math.max(0.5, saved.brushRadius));
		if (typeof saved.protectLabels === "boolean") this.protectLabels = saved.protectLabels;
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
				showLabels: this.showLabels,
				layout: this.layout,
				brushRadius: this.brushRadius,
				protectLabels: this.protectLabels,
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

	zoomBy(factor: number): void {
		this.zoom = Math.min(64, Math.max(1 / 512, this.zoom * factor));
	}

	nextLayout(): void {
		this.layout = LAYOUTS[(LAYOUTS.indexOf(this.layout) + 1) % LAYOUTS.length]!;
	}
}
