<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import { CLOSE_PIXELS, closesAt, closingMode, DRAG_PIXELS, lassoPoints, type Point, type Scale } from "../labels/polygon";
	import { PlaneMask } from "../labels/raster";
	import type { Box, Roi } from "../rois.svelte";
	import type { Stroke } from "./state.svelte";
	import { BACKGROUND_VALUE, withBackground } from "./background";
	import { Busy, type ChunkStore } from "./chunks";
	import { isRightClick } from "./keymap";
	import type { LabelLayer } from "./labels";
	import { MAX_LABEL_TILES, labelsView, overlayHidden } from "./overlays";
	import { type LabelTile, type Overlay, PlaneRenderer } from "./plane";
	import type { ViewerState } from "./state.svelte";
	import {
		boxOnPlane,
		type Level,
		type Plane,
		pixelsPerVoxel,
		type Rect,
		sliceIndex,
		type TileKey,
		tileCrosses,
		tileId,
		tilesToLoad,
		tilesUnder,
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

	// More image chunks than this drawn (the level's and the coarser ones
	// under it), and the view uses a coarser level (each chunk is 64³ voxels,
	// so this bounds the memory a view keeps on big screens).
	const MAX_IMAGE_TILES = 400;
	// How long after the store gave up waiting for a label chunk the server is
	// still making the view asks for it again.
	const UNFINISHED_RETRY_MS = 5000;
	const AXIS_NAMES = ["z", "y", "x"];

	let canvas: HTMLCanvasElement;
	let renderer: PlaneRenderer | undefined;
	let width = $state(0);
	let height = $state(0);
	let error = $state("");
	// Why chunks didn't load, until one does.
	let loadError = $state("");
	// Chunks of the labels the server was still making when the store gave up
	// waiting for them: not an error. The view asks for them again every so
	// often, and says so meanwhile.
	const unfinished = new Set<string>();
	let labelsUnfinished = $state(false);
	let unfinishedTimer: ReturnType<typeof setTimeout> | undefined;
	// Layers left out because zooming out put too many chunks in view: the
	// labels (more than their coarsest level lets them be), and a prediction
	// or segmentation (which have full resolution only).
	let labelsHidden = $state(false);
	let overlaysHidden = $state(false);
	// The level this view showed last, which it keeps a little longer as it
	// zooms in, so it doesn't flip between two levels; the image's, and the labels'.
	let shownLevel: number | undefined;
	let shownLabelLevel: number | undefined;
	let frame = 0;
	let destroyed = false;

	const slice = $derived(Math.floor(viewer.position[plane.normal]));
	const list = new Intl.ListFormat("en", { style: "long", type: "conjunction" });
	const hiddenLayers = $derived([
		...(labelsHidden ? ["labels"] : []),
		...(overlaysHidden && viewer.showPrediction && (prediction || proposal) ? ["the prediction"] : []),
		...(overlaysHidden && viewer.showSegmentation && segmentation ? ["the segmentation"] : []),
	]);

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
		clearTimeout(unfinishedTimer);
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
		const stopClasses = labels.onClasses(() => {
			renderer?.setPalette(labels.colors);
			schedule();
		});
		const stop = labels.onChange((ids) => {
			for (const id of ids) renderer?.dropLabels(id);
			schedule();
		});
		// The server turned out to lack coarser levels the labels listed.
		const stopLevels = labels.onLevels(schedule);
		return () => {
			stopClasses();
			stop();
			stopLevels();
		};
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

	/** The store gave up waiting for a label chunk the server is still making: say so, and have the next frame ask again soon. */
	function stillMaking(id: string) {
		unfinished.add(id);
		labelsUnfinished = true;
		clearTimeout(unfinishedTimer);
		unfinishedTimer = setTimeout(schedule, UNFINISHED_RETRY_MS);
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
		// wanted, and the finer chunks loaded and not yet on the GPU that
		// fill in under the level's missing ones.
		const fine = finer ? sliceIndex(finer, current) : 0;
		const fillers = finer ? underMissing(finer, top, current).filter((key) => images.peek(tileId(key)) && !renderer!.hasImage(key, fine)) : [];
		renderer.reserve(wanted.length + fillers.length, 4 * MAX_LABEL_TILES);
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
		// A prediction or segmentation has full resolution only, so zoomed
		// out far enough it's left out. Only layers that exist and are shown are.
		const predicted = viewer.showPrediction && !!(prediction || proposal);
		const segmented = viewer.showSegmentation && !!segmentation;
		overlaysHidden = (predicted || segmented) && overlayHidden(full, current, overlaysHidden);
		const fullTiles = overlaysHidden ? [] : visibleTiles(full, current, 0);
		const overlays: Overlay[] = [];
		/** Draw a layer's chunks in view (or just `keys`), leaving out `hole`. */
		const add = (
			store: ChunkStore | null | undefined,
			prefix: string,
			shown: boolean,
			opacity: number,
			{ keys = fullTiles, hole, background }: { keys?: TileKey[]; hole?: Rect; background?: boolean } = {},
		) => {
			if (!store) return;
			if (!shown || keys.length === 0) return store.want(plane.name, []);
			const tiles = keys.map((key) => ({ id: `${key.cz}/${key.cy}/${key.cx}`, key }));
			store.want(plane.name, tiles.map((t) => t.id));
			for (const tile of tiles) loadOverlay(store, prefix, tile, levels, at);
			overlays.push({ slice: at, tiles: tiles.map((t) => ({ ...t, id: prefix + t.id })), opacity, hole, background });
		};
		// A proposal shows in its box, in place of the prediction there.
		const inlay = proposal ? boxOnPlane(proposal.box, plane, at) : null;
		add(prediction, "prediction/", viewer.showPrediction, viewer.predictionOpacity, { hole: inlay ?? undefined });
		add(proposal?.store, "proposal/", viewer.showPrediction, viewer.predictionOpacity, {
			keys: inlay ? fullTiles.filter((key) => tileCrosses(key, plane, inlay)) : [],
		});
		add(segmentation, "segmentation/", viewer.showSegmentation, viewer.segmentationOpacity);
		// The labels come at the level that fits the view, with the coarser
		// levels under it, and this page's latest edits over them, so a stroke
		// doesn't vanish as it's finished however far out the view is.
		if (labels && viewer.showLabels) {
			const picked = labelsView(labels.levels, current, {
				current: shownLabelLevel,
				hidden: labelsHidden,
				recent: labels.recent,
				cached: labels.store.ids(),
			});
			shownLabelLevel = picked.level.index;
			labelsHidden = picked.hidden;
			labels.store.want(plane.name, picked.wanted, picked.shown);
			for (const tile of picked.draw) loadOverlay(labels.store, "", tile, labels.levels, at);
			overlays.push({ slice: at, tiles: picked.draw, opacity: viewer.opacity, background: true, level: picked.level.index });
			if (unfinished.size > 0) {
				// Not waiting for chunks the view no longer needs.
				const wanted = new Set(picked.wanted);
				for (const id of unfinished) if (!wanted.has(id)) unfinished.delete(id);
				labelsUnfinished = unfinished.size > 0;
			}
		} else {
			labels?.store.want(plane.name, []);
			labelsHidden = false;
			unfinished.clear();
			labelsUnfinished = false;
		}
		renderer.draw(current, viewer.window, layers, overlays);
	}

	/**
	 * The chunks of `finer` the view shows under the chosen level's chunks
	 * that can't be drawn this frame (not on the GPU, and not loaded to put
	 * there).
	 */
	function underMissing(finer: Level, top: { level: Level; slice: number; tiles: TileKey[] }, current: View): TileKey[] {
		const missing = new Set(top.tiles.filter((key) => !renderer!.hasImage(key, top.slice) && !images.peek(tileId(key))).map(tileId));
		return missing.size === 0 ? [] : tilesUnder(finer, top.level, missing, current);
	}

	/**
	 * Chunks of `finer` already at hand (on the GPU, or loaded) where the
	 * chosen level's chunks haven't arrived, to draw under them meanwhile.
	 */
	function stopgaps(finer: Level, top: { level: Level; slice: number; tiles: TileKey[] }, current: View): TileKey[] {
		if (!renderer) return [];
		const at = sliceIndex(finer, current);
		return underMissing(finer, top, current).filter((key) => {
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

	/**
	 * Load an overlay chunk, or take it from the store if it's there, and put
	 * its slice on the GPU; its textures are named `prefix` + its id. `of` are
	 * the levels its chunk key's level indexes, and `at` the slice of level 0
	 * (`tile` may have a slice of its level's own).
	 */
	function loadOverlay(store: ChunkStore, prefix: string, tile: LabelTile, of: Level[], at: number) {
		const level = of[tile.key.level];
		const slice = tile.slice ?? at;
		const named = { ...tile, id: prefix + tile.id };
		if (!renderer || !level || renderer.hasLabels(named.id, slice)) return;
		const cached = store.get(tile.id);
		if (cached) return renderer.uploadLabels(named, slice, cached);
		if (waiting.has(`overlay:${named.id}`)) return;
		waiting.add(`overlay:${named.id}`);
		store
			.request(tile.id)
			.then((chunk) => {
				// The layer may show another store by now, under the same names.
				if (![prediction, proposal?.store, segmentation, labels?.store].includes(store)) return schedule();
				if (store === labels?.store && unfinished.delete(tile.id)) labelsUnfinished = unfinished.size > 0;
				if (!renderer || renderer.hasLabels(named.id, slice)) return;
				if (sliceIndex(level, view()) !== slice) return schedule();
				renderer.uploadLabels(named, slice, chunk);
				schedule();
			}, (e: unknown) => {
				// A label chunk the server is still making isn't an error.
				if (!(e instanceof Busy)) failed(e);
				else if (store === labels?.store) stillMaking(tile.id);
			})
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

	/** The label values the project can label with: background (1) and its classes. */
	const labelValues = () => [BACKGROUND_VALUE, ...(labels?.classes ?? []).map((c) => c.value)];

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
				onlyIf: viewer.tool === "eraser" ? viewer.eraseCondition(labelValues()) : viewer.paintCondition(labelValues()),
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
			: (withBackground(labels?.classes ?? []).find((c) => c.value === viewer.activeClass)?.color ?? "#ffffff"),
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
		{#if hiddenLayers.length > 0}<span class="muted">· zoom in to see {list.format(hiddenLayers)}</span>{/if}
		{#if labelsUnfinished}<span class="muted">· the server is still making the zoomed-out labels, retrying</span>{/if}
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
