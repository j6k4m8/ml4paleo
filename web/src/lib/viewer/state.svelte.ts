/**
 * What the views of one image share: the crosshair position, the zoom, the
 * display window, and the label overlay. The display preferences persist in
 * this browser.
 */

import type { PlaneMask } from "../labels/raster";
import type { Plane, Vec3 } from "./tiles";

export type Layout = "four" | Plane["name"];
export const LAYOUTS: Layout[] = ["four", "xy", "xz", "yz"];

export type Tool = "navigate" | "brush" | "eraser" | "polygon";

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

/** A polygon being drawn: its plane, slice, and vertices in plane voxels. */
export interface Polygon {
	plane: Plane["name"];
	slice: number;
	points: [number, number][];
}

const PREFERENCES = "m4p.viewer";

interface Preferences {
	opacity: number;
	showLabels: boolean;
	layout: Layout;
	brushRadius: number;
	protectLabels: boolean;
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
	polygon = $state<Polygon | null>(null);
	/** Space is held: drag pans whatever the tool. */
	panning = $state(false);

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
	}

	savePreferences(): void {
		try {
			const preferences: Preferences = {
				opacity: this.opacity,
				showLabels: this.showLabels,
				layout: this.layout,
				brushRadius: this.brushRadius,
				protectLabels: this.protectLabels,
			};
			localStorage.setItem(PREFERENCES, JSON.stringify(preferences));
		} catch {
			// Private windows and blocked storage just don't remember.
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
