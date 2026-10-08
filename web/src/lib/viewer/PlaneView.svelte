<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import { CLOSE_PIXELS, closesAt, closingMode, DRAG_PIXELS, lassoPoints, type Point, type Scale } from "../labels/polygon";
	import { PlaneMask } from "../labels/raster";
	import type { Box, Roi } from "../rois.svelte";
	import type { Stroke } from "./state.svelte";
	import type { ChunkStore } from "./chunks";
	import { isRightClick } from "./keymap";
	import type { LabelLayer } from "./labels";
	import { type LabelTile, type Overlay, PlaneRenderer } from "./plane";
	import type { ViewerState } from "./state.svelte";
	import {
		boxOnPlane,
		countTiles,
		coveringTile,
		type Level,
		type Plane,
		pixelsPerVoxel,
		type Rect,
		sliceIndex,
		type TileKey,
		tileCrosses,
		tileId,
		tilesShown,
		tilesToLoad,
		type Vec3,
		type View,
		viewLevel,
		visibleTiles,
		voxelAt,
	} from "./tiles";

	let {
		plane,
		viewer,
		levels,
		images,
		labels,
		prediction = null,
		proposal = null,
		segmentation = null,
		onhover,
		onresize,
		onstroke,
		onpolygon,
		rois,
		onroi,
	}: {
		plane: Plane;
		viewer: ViewerState;
		levels: Level[];
		images: ChunkStore;
		labels: LabelLayer | null;
		/** A model's prediction (label values), drawn under the labels. */
		prediction?: ChunkStore | null;
		/**
		 * A proposal: one box predicted on demand (empty elsewhere), drawn in
		 * place of the prediction there, with the prediction's visibility.
		 */
		proposal?: { store: ChunkStore; box: Box } | null;
		/** The final segmentation (label values), drawn over the prediction. */
		segmentation?: ChunkStore | null;
		onhover: (plane: Plane) => void;
		onresize: (plane: Plane, width: number, height: number) => void;
		/** A finished brush or eraser stroke, with the settings it began with. */
		onstroke: (stroke: Stroke) => void;
		/** Close the polygon being drawn: fill it, or with `cut`, cut it out of the active class. */
		onpolygon: (cut: boolean) => void;
		rois: Roi[];
		/** A rectangle drawn with the ROI tool, in plane voxels, at `slice`. */
		onroi: (plane: Plane, slice: number, corners: [[number, number], [number, number]]) => void;
	} = $props();

	// Labels are full resolution only until the label pyramid exists, so
	// zoomed far out they'd need too many chunks (each is 256 KiB; three
	// views share a 128 MiB cache).
	const MAX_LABEL_TILES = 128;
	// Once hidden, overlays show again only when the view needs this many
	// times fewer chunks, so zooming near the limit doesn't flicker them.
	const OVERLAY_HYSTERESIS = 1.25;
	// More image chunks than this drawn (the level's and the coarser ones
	// under it), and the view uses a coarser level (each chunk is 64³ voxels,
	// so this bounds the memory a view keeps on big screens).
	const MAX_IMAGE_TILES = 400;
	const AXIS_NAMES = ["z", "y", "x"];

	let canvas: HTMLCanvasElement;
	let renderer: PlaneRenderer | undefined;
	let width = $state(0);
	let height = $state(0);
	let error = $state("");
	// Why chunks didn't load, until one does.
	let loadError = $state("");
	let labelsHidden = $state(false);
	// The level this view showed last, which it keeps a little longer as it
	// zooms in, so it doesn't flip between two levels.
	let shownLevel: number | undefined;
	let frame = 0;
	let destroyed = false;

	const slice = $derived(Math.floor(viewer.position[plane.normal]));

	// Loads this view is waiting for, so each gets one handler.
	const waiting = new Set<string>();

	function view(): View {
		return {
			plane,
			position: viewer.position,
			zoom: viewer.zoom,
			aspect: viewer.aspect,
			width,
			height,
			pixelRatio: window.devicePixelRatio || 1,
		};
	}

	function start() {
		renderer = new PlaneRenderer(canvas, plane, viewer.shape);
		if (labels) renderer.setPalette(labels.colors);
	}

	onMount(() => {
		const observer = new ResizeObserver(([entry]) => {
			if (!entry) return;
			const ratio = window.devicePixelRatio || 1;
			width = Math.max(1, Math.round(entry.contentRect.width * ratio));
			height = Math.max(1, Math.round(entry.contentRect.height * ratio));
			canvas.width = width;
			canvas.height = height;
			onresize(plane, width, height);
		});
		observer.observe(canvas);
		// The browser may drop the GPU context (driver reset, too many
		// contexts); start over with a fresh renderer when it comes back.
		const lost = (event: Event) => {
			event.preventDefault();
			renderer = undefined;
			waiting.clear();
		};
		const restored = () => {
			start();
			schedule();
		};
		canvas.addEventListener("webglcontextlost", lost);
		canvas.addEventListener("webglcontextrestored", restored);
		try {
			start();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
		return () => {
			observer.disconnect();
			canvas.removeEventListener("webglcontextlost", lost);
			canvas.removeEventListener("webglcontextrestored", restored);
		};
	});

	onDestroy(() => {
		destroyed = true;
		cancelAnimationFrame(frame);
		images.want(plane.name, new Set());
		labels?.store.want(plane.name, new Set());
		prediction?.want(plane.name, new Set());
		proposal?.store.want(plane.name, new Set());
		segmentation?.want(plane.name, new Set());
		renderer?.destroy();
		renderer = undefined;
	});

	// Switching tools (Esc, a key) drops an ROI being drawn.
	$effect(() => {
		if (viewer.tool !== "roi") rectangle = null;
	});

	// A layer that now shows another store (a newer prediction or proposal,
	// say) has other contents: forget the old one's textures.
	let shownStores: Record<string, ChunkStore | null | undefined> = {};
	$effect(() => {
		const stores = { "prediction/": prediction, "proposal/": proposal?.store, "segmentation/": segmentation };
		for (const [prefix, store] of Object.entries(stores)) {
			if (prefix in shownStores && shownStores[prefix] !== store) {
				shownStores[prefix]?.want(plane.name, new Set());
				renderer?.dropOverlay(prefix);
				schedule();
			}
		}
		shownStores = stores;
	});

	// New label colors, and chunks someone else just edited.
	$effect(() => {
		if (!labels) return;
		renderer?.setPalette(labels.colors);
		const stop = labels.onChange((ids) => {
			for (const id of ids) renderer?.dropLabels(id);
			schedule();
		});
		return stop;
	});

	$effect(() => {
		// Redraw whenever anything the view shows changes.
		void [
			viewer.position,
			viewer.zoom,
			viewer.window[0],
			viewer.window[1],
			viewer.opacity,
			viewer.showLabels,
			viewer.showPrediction,
			viewer.predictionOpacity,
			viewer.showSegmentation,
			viewer.segmentationOpacity,
			width,
			height,
			labels,
			prediction,
			proposal,
			segmentation,
		];
		schedule();
	});

	function schedule() {
		if (destroyed) return;
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(render);
	}

	function failed(e: unknown) {
		if (e instanceof DOMException && e.name === "AbortError") return;
		loadError = e instanceof Error ? e.message : String(e);
	}

	function render() {
		if (destroyed || !renderer || levels.length === 0 || width === 0) return;
		const current = view();
		const chosen = viewLevel(levels, current, shownLevel, MAX_IMAGE_TILES);
		shownLevel = chosen.index;
		const slices = levels.map((level) => sliceIndex(level, current));
		// Every coarser level loads first, where the view shows it, and stays
		// drawn under the finer ones: whatever moves (zoom, pan, slice), the
		// view shows the best it has while the rest arrives, never a gap.
		// What it draws stays cached; the margin loads ahead but may go.
		const { tiles: wanted, shown } = tilesToLoad(levels, chosen, current);
		images.want(plane.name, wanted.map(tileId), wanted.slice(0, shown).map(tileId));
		const top = { level: chosen, slice: slices[chosen.index]!, tiles: visibleTiles(chosen, current, 0) };
		const finer = levels[chosen.index - 1];
		// Room on the GPU for every texture this frame may upload, made before
		// any upload, so none pushes out another the frame draws: the chunks
		// wanted, and the finer chunks under each of the level's still missing.
		const under = finer ? (chosen.scale[plane.u] / finer.scale[plane.u]) * (chosen.scale[plane.v] / finer.scale[plane.v]) : 0;
		const missing = under ? top.tiles.filter((key) => !renderer!.hasImage(key, top.slice)).length : 0;
		renderer.reserve(wanted.length + missing * under, 4 * MAX_LABEL_TILES);
		for (const key of wanted) loadImage(key, slices[key.level]!);
		const layers: { level: Level; slice: number; tiles: TileKey[] }[] = [];
		for (let i = levels.length - 1; i > chosen.index; i--) {
			layers.push({ level: levels[i]!, slice: slices[i]!, tiles: visibleTiles(levels[i]!, current, 0) });
		}
		// Zooming out, what the finer level already has fills in until the
		// chosen level's chunks arrive.
		const filling = finer ? { level: finer, slice: slices[finer.index]!, tiles: stopgaps(finer, top, current) } : null;
		if (filling && filling.tiles.length > 0) layers.push(filling);
		layers.push(top);

		const full = levels[0]!;
		const at = slices[0]!;
		// Only layers that exist and are shown get hidden when zoomed out.
		const overlaid = !!(
			(viewer.showLabels && labels) ||
			(viewer.showPrediction && (prediction || proposal)) ||
			(viewer.showSegmentation && segmentation)
		);
		const needed = countTiles(full, current);
		labelsHidden = overlaid && needed > (labelsHidden ? MAX_LABEL_TILES / OVERLAY_HYSTERESIS : MAX_LABEL_TILES);
		const fullTiles = labelsHidden ? [] : visibleTiles(full, current, 0);
		const overlays: Overlay[] = [];
		/** Draw a layer's chunks in view (or just `keys`), leaving out `hole`. */
		const add = (
			store: ChunkStore | null | undefined,
			prefix: string,
			shown: boolean,
			opacity: number,
			{ keys = fullTiles, hole }: { keys?: TileKey[]; hole?: Rect } = {},
		) => {
			if (!store) return;
			if (!shown || keys.length === 0) return store.want(plane.name, []);
			const tiles = keys.map((key) => ({ id: `${key.cz}/${key.cy}/${key.cx}`, key }));
			store.want(plane.name, tiles.map((t) => t.id));
			for (const tile of tiles) loadOverlay(store, prefix, tile, at);
			overlays.push({ slice: at, tiles: tiles.map((t) => ({ ...t, id: prefix + t.id })), opacity, hole });
		};
		// A proposal shows in its box, in place of the prediction there.
		const inlay = proposal ? boxOnPlane(proposal.box, plane, at) : null;
		add(prediction, "prediction/", viewer.showPrediction, viewer.predictionOpacity, { hole: inlay ?? undefined });
		add(proposal?.store, "proposal/", viewer.showPrediction, viewer.predictionOpacity, {
			keys: inlay ? fullTiles.filter((key) => tileCrosses(key, plane, inlay)) : [],
		});
		add(segmentation, "segmentation/", viewer.showSegmentation, viewer.segmentationOpacity);
		// Zoomed out past the limit, this page's latest edits still show, so a
		// stroke doesn't vanish as it's finished.
		const edited = labelsHidden && labels ? tilesShown(labels.recent.map(labelTile), full, current).slice(0, MAX_LABEL_TILES) : fullTiles;
		add(labels?.store, "", viewer.showLabels, viewer.opacity, { keys: edited });
		renderer.draw(current, viewer.window, layers, overlays);
	}

	/** The level-0 chunk key of a label chunk id (`cz/cy/cx`). */
	function labelTile(id: string): TileKey {
		const [cz = 0, cy = 0, cx = 0] = id.split("/").map(Number);
		return { level: 0, cz, cy, cx };
	}

	/**
	 * Chunks of `finer` already at hand (on the GPU, or loaded) where the
	 * chosen level's chunks haven't arrived, to draw under them meanwhile.
	 */
	function stopgaps(finer: Level, top: { level: Level; slice: number; tiles: TileKey[] }, current: View): TileKey[] {
		if (!renderer) return [];
		const missing = new Set(top.tiles.filter((key) => !renderer!.hasImage(key, top.slice)).map(tileId));
		if (missing.size === 0) return [];
		const at = sliceIndex(finer, current);
		return visibleTiles(finer, current, 0).filter((key) => {
			if (!missing.has(tileId(coveringTile(key, finer, top.level, current)))) return false;
			if (renderer!.hasImage(key, at)) return true;
			const cached = images.peek(tileId(key));
			if (cached) renderer!.uploadImage(key, at, cached);
			return !!cached;
		});
	}

	function loadImage(key: TileKey, at: number) {
		if (!renderer || renderer.hasImage(key, at)) return;
		const id = tileId(key);
		const cached = images.get(id);
		if (cached) return renderer.uploadImage(key, at, cached);
		if (waiting.has(`image:${id}`)) return;
		waiting.add(`image:${id}`);
		images
			.request(id)
			.then((chunk) => {
				loadError = "";
				if (!renderer || renderer.hasImage(key, at)) return;
				if (sliceIndex(levels[key.level]!, view()) !== at) return schedule();
				renderer.uploadImage(key, at, chunk);
				schedule();
			}, failed)
			.finally(() => waiting.delete(`image:${id}`));
	}

	/** Load an overlay chunk; its textures are named `prefix` + its id. */
	function loadOverlay(store: ChunkStore, prefix: string, tile: LabelTile, at: number) {
		const named = { ...tile, id: prefix + tile.id };
		if (!renderer || renderer.hasLabels(named.id, at)) return;
		const cached = store.get(tile.id);
		if (cached) return renderer.uploadLabels(named, at, cached);
		if (waiting.has(`overlay:${named.id}`)) return;
		waiting.add(`overlay:${named.id}`);
		store
			.request(tile.id)
			.then((chunk) => {
				// The layer may show another store by now, under the same names.
				if (![prediction, proposal?.store, segmentation, labels?.store].includes(store)) return schedule();
				if (!renderer || renderer.hasLabels(named.id, at)) return;
				if (sliceIndex(levels[0]!, view()) !== at) return schedule();
				renderer.uploadLabels(named, at, chunk);
				schedule();
			}, failed)
			.finally(() => waiting.delete(`overlay:${named.id}`));
	}

	// --- pointer and wheel ---------------------------------------------------

	// `draws`: the press added a polygon point, so dragging on draws freehand;
	// `before`: how many points the polygon had before the press.
	let press: { x: number; y: number; moved: boolean; pan: boolean; pointer: number; draws: boolean; before: number } | null = null;
	let stroke: (Stroke & { last: [number, number] }) | null = null;
	let rectangle: { from: [number, number]; to: [number, number] } | null = $state(null);
	let cursor: [number, number] | null = $state(null);
	let overlay: HTMLCanvasElement;
	let wheelSteps = 0;

	const ratio = () => window.devicePixelRatio || 1;
	const painting = $derived(viewer.tool === "brush" || viewer.tool === "eraser");

	function offset(event: MouseEvent): [number, number] {
		const rect = canvas.getBoundingClientRect();
		return [(event.clientX - rect.left) * ratio() - width / 2, (event.clientY - rect.top) * ratio() - height / 2];
	}

	/** The plane point (u, v) under the pointer, in level-0 voxels. */
	function planePoint(event: MouseEvent): [number, number] {
		const point = voxelAt(view(), ...offset(event));
		return [point[plane.u], point[plane.v]];
	}

	/** The plane point at a place on the view, in CSS pixels from its corner. */
	function planeAt([x, y]: [number, number]): Point {
		const point = voxelAt(view(), x * ratio() - width / 2, y * ratio() - height / 2);
		return [point[plane.u], point[plane.v]];
	}

	/** CSS pixels per level-0 voxel along u and v. */
	function scale(): Scale {
		const px = pixelsPerVoxel(view());
		return [px[plane.u] / ratio(), px[plane.v] / ratio()];
	}

	/** Fingers and pens are less exact than a mouse, so they get twice the room. */
	const reach = (event: PointerEvent) => (event.pointerType === "mouse" ? 1 : 2);

	/** Whether closing the polygon with this event's keys held cuts it out. */
	const cuts = (event: MouseEvent) => closingMode(viewer.polygonMode, event) === "subtract";

	/** Where a plane point is on screen, in CSS pixels. */
	function screen(u: number, v: number): [number, number] {
		const px = pixelsPerVoxel(view());
		return [
			((u - viewer.position[plane.u]) * px[plane.u] + width / 2) / ratio(),
			((v - viewer.position[plane.v]) * px[plane.v] + height / 2) / ratio(),
		];
	}

	/** Brush radii along u and v, in level-0 voxels. */
	function radii(): [number, number] {
		return [viewer.brushRadius / viewer.aspect[plane.u], viewer.brushRadius / viewer.aspect[plane.v]];
	}

	function canEdit(): boolean {
		return viewer.tool === "eraser" || viewer.tool === "roi" || viewer.activeClass !== null;
	}

	function pointerDown(event: PointerEvent) {
		// One pointer at a time: a second finger doesn't join the stroke.
		if (press) return;
		canvas.focus();
		viewer.noteKeys(event);
		// Right-click goes to the spot, with any tool; a plain click never moves you.
		if (isRightClick(event)) return goHere(event);
		canvas.setPointerCapture(event.pointerId);
		const pan = viewer.tool === "navigate" || viewer.panning || event.button === 1;
		press = { x: event.clientX, y: event.clientY, moved: false, pan, pointer: event.pointerId, draws: false, before: 0 };
		if (pan || event.button !== 0 || !canEdit()) return;
		if (painting) {
			const point = planePoint(event);
			const mask = new PlaneMask(viewer.shape[plane.u], viewer.shape[plane.v]);
			mask.stamp(...point, ...radii());
			// The slice, tool, and class are the ones the stroke started with.
			stroke = {
				plane,
				slice,
				mask,
				erase: viewer.tool === "eraser",
				value: viewer.tool === "eraser" ? 0 : (viewer.activeClass ?? 0),
				onlyIf: viewer.tool === "eraser" || !viewer.protectLabels ? "any" : "unlabeled",
				radius: viewer.brushRadius,
				last: point,
			};
			drawStroke();
		} else if (viewer.tool === "roi") {
			const point = planePoint(event);
			rectangle = { from: point, to: point };
		} else if (viewer.tool === "polygon") {
			const point = planePoint(event);
			const current = polygonHere;
			// Clicking the first point again closes the polygon.
			if (current && closesAt(current.points, point, scale(), CLOSE_PIXELS * reach(event))) return onpolygon(cuts(event));
			viewer.polygon = current ? { ...current, points: [...current.points, point] } : { plane: plane.name, slice, points: [point] };
			press.draws = true;
			press.before = current?.points.length ?? 0;
		}
	}

	function pointerMove(event: PointerEvent) {
		onhover(plane);
		if (press && event.pointerId !== press.pointer) return;
		viewer.noteKeys(event);
		cursor = offset(event).map((d, i) => (d + (i === 0 ? width : height) / 2) / ratio()) as [number, number];
		if (rectangle) {
			if (press && Math.hypot(event.clientX - press.x, event.clientY - press.y) >= 3) press.moved = true;
			rectangle = { ...rectangle, to: planePoint(event) };
			return;
		}
		if (stroke) {
			const point = planePoint(event);
			stroke.mask.line(stroke.last, point, ...radii());
			stroke.last = point;
			drawStroke();
			return;
		}
		if (press?.draws) return lasso(press, event);
		if (!press?.pan) return;
		const dx = event.clientX - press.x;
		const dy = event.clientY - press.y;
		if (!press.moved && Math.hypot(dx, dy) < 3 * reach(event)) return;
		press.moved = true;
		viewer.autoFit = false;
		const px = pixelsPerVoxel(view());
		const point = [...viewer.position] as Vec3;
		point[plane.u] -= (dx * ratio()) / px[plane.u];
		point[plane.v] -= (dy * ratio()) / px[plane.v];
		viewer.moveTo(point);
		press.x = event.clientX;
		press.y = event.clientY;
	}

	/** Follow a drag with the polygon tool, dropping points along the way. */
	function lasso(drawing: NonNullable<typeof press>, event: PointerEvent) {
		const current = polygonHere;
		if (!current) {
			// Closed or dropped meanwhile (Enter, Esc, another tool).
			drawing.draws = false;
			viewer.lassoing = false;
			return;
		}
		if (!viewer.lassoing && Math.hypot(event.clientX - drawing.x, event.clientY - drawing.y) < DRAG_PIXELS * reach(event)) return;
		viewer.lassoing = true;
		// Moves the browser merged into this one keep fast drags smooth.
		const merged = event.getCoalescedEvents?.() ?? [];
		const samples = (merged.length > 0 ? merged : [event]).map(planePoint);
		const added = lassoPoints(current.points.at(-1), samples, scale());
		if (added.length > 0) viewer.polygon = { ...current, points: [...current.points, ...added] };
	}

	function pointerUp(event: PointerEvent) {
		if (press && event.pointerId !== press.pointer) return;
		if (rectangle) {
			const { from, to } = rectangle;
			rectangle = null;
			// A click without a drag isn't an ROI.
			if (press?.moved) onroi(plane, slice, [from, to]);
		} else if (stroke) {
			const { last: _last, ...finished } = stroke;
			stroke = null;
			clearStroke();
			if (finished.mask.count > 0) onstroke(finished);
		} else if (press?.draws && viewer.lassoing) {
			// Letting go of a freehand drag closes it, where the pointer let go.
			viewer.lassoing = false;
			const current = polygonHere;
			if (current) {
				const end = planePoint(event);
				const last = current.points.at(-1);
				if (!last || last[0] !== end[0] || last[1] !== end[1]) viewer.polygon = { ...current, points: [...current.points, end] };
				onpolygon(cuts(event));
			}
		} else if (press?.pan && !press.moved && viewer.tool === "navigate" && event.pointerType !== "mouse") {
			// A tap with a finger or a pen, which may have no right button.
			goHere(event);
		}
		press = null;
	}

	/** Move the crosshair to where the pointer is. */
	function goHere(event: PointerEvent) {
		viewer.autoFit = false;
		viewer.moveTo(voxelAt(view(), ...offset(event)));
	}

	function cancel() {
		// A drag the browser called off (for a system gesture, say) takes back the points it added.
		if (press?.draws && viewer.lassoing && polygonHere) {
			viewer.polygon = press.before > 0 ? { ...polygonHere, points: polygonHere.points.slice(0, press.before) } : null;
		}
		if (press?.draws) viewer.lassoing = false;
		press = null;
		stroke = null;
		rectangle = null;
		clearStroke();
	}

	function wheel(event: WheelEvent) {
		event.preventDefault();
		const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
		if (event.ctrlKey || event.metaKey) {
			viewer.autoFit = false;
			// Keep the voxel under the cursor in place.
			const [dx, dy] = offset(event);
			const under = voxelAt(view(), dx, dy);
			viewer.zoomBy(Math.exp(-delta * 0.002));
			const px = pixelsPerVoxel(view());
			const point = [...viewer.position] as Vec3;
			point[plane.u] = under[plane.u] - dx / px[plane.u];
			point[plane.v] = under[plane.v] - dy / px[plane.v];
			viewer.moveTo(point);
			return;
		}
		if (stroke || rectangle || viewer.lassoing) return;
		// About one slice per mouse wheel notch; trackpads add up.
		wheelSteps += delta / 100;
		const steps = Math.trunc(wheelSteps);
		if (steps !== 0) {
			wheelSteps -= steps;
			viewer.step(plane.normal, steps);
		}
	}

	// --- previews --------------------------------------------------------------

	const activeColor = $derived(
		viewer.tool === "eraser"
			? "#ffffff"
			: (labels?.classes.find((c) => c.value === viewer.activeClass)?.color ?? "#ffffff"),
	);

	function clearStroke() {
		overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
	}

	/** Paint the stroke so far: one rectangle per run of voxels in a row. */
	function drawStroke() {
		if (!stroke || !overlay) return;
		overlay.width = width;
		overlay.height = height;
		const context = overlay.getContext("2d");
		if (!context) return;
		const { mask } = stroke;
		const px = pixelsPerVoxel(view());
		const r = ratio();
		context.globalAlpha = Math.max(0.35, viewer.opacity);
		context.fillStyle = activeColor;
		for (let j = mask.v0; j < mask.v0 + mask.height; j++) {
			let i = mask.u0;
			while (i < mask.u0 + mask.width) {
				if (!mask.has(i, j)) {
					i++;
					continue;
				}
				let end = i;
				while (end < mask.u0 + mask.width && mask.has(end, j)) end++;
				const [x, y] = screen(i, j);
				context.fillRect(x * r, y * r, (end - i) * px[plane.u], px[plane.v]);
				i = end;
			}
		}
	}

	const polygonHere = $derived(
		viewer.polygon && viewer.polygon.plane === plane.name && viewer.polygon.slice === slice ? viewer.polygon : null,
	);

	// Closing the polygon now would cut it out of the active class.
	const cutting = $derived(viewer.tool === "polygon" && closingMode(viewer.polygonMode, viewer.held) === "subtract");

	/**
	 * The polygon being drawn here, on screen: its outline so far (to the
	 * pointer), its first point once a click there can close it, and whether
	 * the pointer is close enough to that point for a click to.
	 */
	const polygonShape = $derived.by(() => {
		void [viewer.position, viewer.zoom, width, height];
		const points = polygonHere?.points ?? [];
		if (points.length === 0) return null;
		const outline = points.map(([u, v]) => screen(u, v));
		const first = outline[0]!;
		const closing = !viewer.lassoing && cursor !== null && closesAt(points, planeAt(cursor), scale());
		// The outline ends at the pointer, or closes when a click or letting go would close it.
		if (cursor && !closing) outline.push(cursor);
		if (closing || viewer.lassoing) outline.push(first);
		return {
			outline: outline.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "),
			first: points.length >= 3 && !viewer.lassoing ? first : null,
			closing,
		};
	});

	/** Screen rectangles of the ROIs this view's plane cuts through. */
	const roiOutlines = $derived.by(() => {
		void [viewer.position, viewer.zoom, width, height];
		const { normal, u, v } = plane;
		return rois
			.filter((roi) => roi.bbox[normal]! <= slice && slice < roi.bbox[normal + 3]!)
			.map((roi) => {
				const [x0, y0] = screen(roi.bbox[u]!, roi.bbox[v]!);
				const [x1, y1] = screen(roi.bbox[u + 3]!, roi.bbox[v + 3]!);
				return { id: roi.id, status: roi.status, x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
			});
	});

	const rectangleOutline = $derived.by(() => {
		void [viewer.position, viewer.zoom, width, height];
		if (!rectangle) return null;
		const [x0, y0] = screen(...rectangle.from);
		const [x1, y1] = screen(...rectangle.to);
		return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
	});

	const brushOutline = $derived.by(() => {
		void [viewer.position, viewer.zoom, viewer.brushRadius, width, height];
		if (!painting || !cursor) return null;
		const px = pixelsPerVoxel(view());
		const [ru, rv] = radii();
		return { x: cursor[0], y: cursor[1], rx: (ru * px[plane.u]) / ratio(), ry: (rv * px[plane.v]) / ratio() };
	});
</script>

<div class="plane plane-{plane.normal}">
	<canvas
		bind:this={canvas}
		tabindex="0"
		aria-label="{plane.name.toUpperCase()} view at {AXIS_NAMES[plane.normal]} {slice}"
		onpointerdown={pointerDown}
		onpointermove={pointerMove}
		onpointerup={pointerUp}
		onpointercancel={cancel}
		onpointerenter={() => onhover(plane)}
		onpointerleave={() => (cursor = null)}
		oncontextmenu={(event) => event.preventDefault()}
		ondblclick={(event) => viewer.tool === "polygon" && onpolygon(cuts(event))}
		onfocus={() => onhover(plane)}
		onwheel={wheel}
		class:editing={viewer.tool !== "navigate" && !viewer.panning}
	></canvas>
	<canvas class="overlay" bind:this={overlay} aria-hidden="true"></canvas>
	<svg class="overlay" aria-hidden="true">
		{#each roiOutlines as roi (roi.id)}
			<rect
				class="roi roi-{roi.status}"
				class:selected={roi.id === viewer.selectedRoi}
				x={roi.x}
				y={roi.y}
				width={Math.max(1, roi.w)}
				height={Math.max(1, roi.h)}
			/>
		{/each}
		{#if rectangleOutline}
			<rect class="roi roi-new" x={rectangleOutline.x} y={rectangleOutline.y} width={rectangleOutline.w} height={rectangleOutline.h} />
		{/if}
		{#if polygonShape}
			<!-- A cutout shows dashed over a darker fill. -->
			<polyline
				points={polygonShape.outline}
				fill={cutting ? "#000000" : activeColor}
				fill-opacity={cutting ? 0.35 : 0.25}
				stroke={activeColor}
				stroke-width="1.5"
				stroke-dasharray={cutting ? "5 3" : undefined}
			/>
			{#if polygonShape.first}
				<!-- Click here to close; it grows and fills once the pointer is close enough. -->
				<circle
					class="start"
					class:closing={polygonShape.closing}
					cx={polygonShape.first[0]}
					cy={polygonShape.first[1]}
					r={polygonShape.closing ? 6 : 3.5}
					stroke={activeColor}
				/>
			{/if}
		{/if}
		{#if cutting && cursor && viewer.activeClass !== null}
			<!-- A minus beside the pointer, as image editors show for subtracting. -->
			<g class="cut-mark" transform="translate({cursor[0] + 11} {cursor[1] + 11})">
				<circle r="6" />
				<line x1="-3" x2="3" y1="0" y2="0" />
			</g>
		{/if}
		{#if brushOutline}
			<ellipse
				cx={brushOutline.x}
				cy={brushOutline.y}
				rx={Math.max(1, brushOutline.rx)}
				ry={Math.max(1, brushOutline.ry)}
				fill="none"
				stroke={activeColor}
				stroke-width="1"
			/>
		{/if}
	</svg>
	<div class="crosshair u axis-{plane.u}" aria-hidden="true"></div>
	<div class="crosshair v axis-{plane.v}" aria-hidden="true"></div>
	<div class="caption">
		{plane.name.toUpperCase()} · {AXIS_NAMES[plane.normal]}
		{slice}
		{#if labels && !viewer.showLabels}<span class="warn">· labels hidden (V)</span>{/if}
		{#if labelsHidden}<span class="muted">· zoom in to see labels</span>{/if}
	</div>
	{#if error || loadError}<p class="error" role="alert">{error || loadError}</p>{/if}
</div>

<style>
	.plane {
		position: relative;
		min-width: 0;
		min-height: 0;
		border-top: 2px solid var(--axis-color);
		background: var(--color-pasteboard);
		overflow: hidden;
	}
	.plane-0 {
		--axis-color: var(--color-axis-z);
	}
	.plane-1 {
		--axis-color: var(--color-axis-y);
	}
	.plane-2 {
		--axis-color: var(--color-axis-x);
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		background: var(--color-pasteboard);
		touch-action: none;
		cursor: grab;
	}
	canvas.editing {
		cursor: crosshair;
	}
	.roi {
		fill: none;
		stroke-width: 1.5;
		stroke-dasharray: 6 3;
	}
	.roi-open {
		stroke: var(--color-warn);
	}
	.roi-complete {
		stroke: var(--color-ok);
		stroke-dasharray: none;
	}
	.roi-skipped {
		stroke: var(--color-ink-faint);
		stroke-dasharray: 2 3;
	}
	.roi-new {
		stroke: #fff;
	}
	.roi.selected {
		stroke-width: 3;
	}
	.start {
		fill: rgb(0 0 0 / 0.6);
		stroke-width: 1.5;
	}
	.start.closing {
		fill: #fff;
		stroke-width: 2;
	}
	.cut-mark circle {
		fill: rgb(0 0 0 / 0.7);
		stroke: #fff;
		stroke-width: 1;
	}
	.cut-mark line {
		stroke: #fff;
		stroke-width: 1.5;
	}
	.overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		background: none;
	}
	canvas:focus-visible {
		outline: 1px solid var(--color-accent);
		outline-offset: -2px;
	}
	.crosshair {
		position: absolute;
		pointer-events: none;
		opacity: 0.6;
	}
	/* The line along u marks the plane of constant v, and vice versa. */
	.crosshair.u {
		left: 50%;
		top: 0;
		bottom: 0;
		width: 1px;
	}
	.crosshair.v {
		top: 50%;
		left: 0;
		right: 0;
		height: 1px;
	}
	.axis-0 {
		background: var(--color-axis-z);
	}
	.axis-1 {
		background: var(--color-axis-y);
	}
	.axis-2 {
		background: var(--color-axis-x);
	}
	.caption {
		position: absolute;
		left: 0.375rem;
		top: 0.375rem;
		padding: 0.0625rem 0.375rem;
		border-radius: 2px;
		background: rgb(0 0 0 / 0.55);
		font-family: var(--font-mono);
		font-size: 0.6875rem;
		color: #e2e2e2;
		pointer-events: none;
	}
	.caption .muted {
		color: #a8a8a8;
	}
	.caption .warn {
		color: var(--color-warn);
	}
	.error {
		position: absolute;
		bottom: 0.375rem;
		left: 0.375rem;
		margin: 0;
		padding: 0.125rem 0.375rem;
		border-radius: 2px;
		background: rgb(0 0 0 / 0.7);
		color: var(--color-danger);
		font-size: 0.6875rem;
	}
</style>
