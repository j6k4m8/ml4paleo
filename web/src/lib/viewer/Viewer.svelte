<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import Check from "@lucide/svelte/icons/check";
	import CheckCheck from "@lucide/svelte/icons/check-check";
	import CloudOff from "@lucide/svelte/icons/cloud-off";
	import Compass from "@lucide/svelte/icons/compass";
	import Eraser from "@lucide/svelte/icons/eraser";
	import Eye from "@lucide/svelte/icons/eye";
	import EyeOff from "@lucide/svelte/icons/eye-off";
	import Hand from "@lucide/svelte/icons/hand";
	import ImageOff from "@lucide/svelte/icons/image-off";
	import Keyboard from "@lucide/svelte/icons/keyboard";
	import LayoutGrid from "@lucide/svelte/icons/layout-grid";
	import LoaderCircle from "@lucide/svelte/icons/loader-circle";
	import Lock from "@lucide/svelte/icons/lock";
	import Maximize2 from "@lucide/svelte/icons/maximize-2";
	import PanelRight from "@lucide/svelte/icons/panel-right";
	import Pentagon from "@lucide/svelte/icons/pentagon";
	import Plus from "@lucide/svelte/icons/plus";
	import Redo2 from "@lucide/svelte/icons/redo-2";
	import RotateCcw from "@lucide/svelte/icons/rotate-ccw";
	import Sparkles from "@lucide/svelte/icons/sparkles";
	import SquareDashed from "@lucide/svelte/icons/square-dashed";
	import SquaresSubtract from "@lucide/svelte/icons/squares-subtract";
	import SquaresUnite from "@lucide/svelte/icons/squares-unite";
	import Undo2 from "@lucide/svelte/icons/undo-2";
	import X from "@lucide/svelte/icons/x";
	import { onDestroy, onMount, tick, untrack } from "svelte";
	import { SvelteMap } from "svelte/reactivity";
	import ClassMenu from "#lib/ui/ClassMenu.svelte";
	import Histogram from "#lib/ui/Histogram.svelte";
	import { displayLevel, editLevel } from "#lib/ui/histogram.ts";
	import Panel from "#lib/ui/Panel.svelte";
	import Segmented from "#lib/ui/Segmented.svelte";
	import SliderField from "#lib/ui/SliderField.svelte";
	import ToolButton from "#lib/ui/ToolButton.svelte";
	import { tooltip } from "#lib/ui/tooltip.ts";
	import { ApiError, api, message } from "#lib/api.ts";
	import { unfinished } from "#lib/pipelines.ts";
	import { whileVisible } from "#lib/refresh.ts";
	import { nextColor } from "#lib/labelimport.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { Pipeline, ProjectImage } from "#lib/types.ts";
	import {
		acceptKeyTarget,
		acceptCounts,
		acceptParts,
		declineParts,
		declinesUnderErase,
		MAX_ACCEPT_VOXELS,
		planeToAccept,
		readBox,
		restoreDeclinedParts,
		unlabeledOnly,
		viewExtent,
		whyNotInView,
	} from "../labels/accept";
	import { splitIntoDeltas } from "../labels/deltas";
	import { acceptGestureGroups, gestureBox, type GestureSource } from "../labels/accept-gesture";
	import { indexedDbStorage, OpQueue, type QueuedEdit, saveState } from "../labels/opqueue.svelte";
	import { ACCEPT_MODES, type AcceptMode, describeWhere, ERASE_MODES, type EraseMode, PAINT_MODES, type PaintMode } from "../labels/modes";
	import { closingMode, type PolygonMode, polygonEdit } from "../labels/polygon";
	import { PlaneMask } from "../labels/raster";
	import {
		type Box,
		clipBox,
		describe,
		overlaps,
		revert,
		type Roi,
		RoiList,
		roiBox,
		thinAxis,
		voxels,
		within,
	} from "../rois.svelte";
	import { SHOW_ROIS } from "../features";
	import { BACKGROUND_VALUE, DECLINED_VALUE, withBackground } from "./background";
	import { ChunkStore } from "./chunks";
	import ClassRow from "./ClassRow.svelte";
	import { ClassDisplay } from "./class-display.svelte";
	import { classOpacity, displayedValues, type ClassStyles } from "./class-display";
	import { absolute, loadLevels } from "./image";
	import { type Action, actionFor, forFocused, KEYMAP, MOUSE } from "./keymap";
	import { type LabelClass, LabelLayer, strictOn } from "./labels";
	import { imageLoader, labelLoader, WorkerPool } from "./loader";
	import PlaneView from "./PlaneView.svelte";
	import MiniMap from "./MiniMap.svelte";
	import { LivePreview } from "./live.svelte";
	import LivePanel from "./LivePanel.svelte";
	import { reviewSlice } from "./live-copy";
	import { intersection, liveKeys, type ModelPlugin } from "./live";
	import { LAYOUTS, type Stroke, ViewerState } from "./state.svelte";
	import { aspectOf, type Level, type Plane, PLANES, type Vec3 } from "./tiles";

	let {
		image,
		projectId,
		roi: startRoi = null,
		box: startBox = null,
	}: { image: ProjectImage; projectId: string; roi?: string | null; box?: Box | null } = $props();

	const CACHE_BYTES = 512 * 1024 * 1024;
	// The most one proposal predicts (the server's limit).
	const MAX_PROPOSAL_VOXELS = 256 ** 3;

	// The page makes a new viewer for each image.
	const { manifest, zarr_url: zarrUrl, artifact_id: imageId } = untrack(() => image);
	const project = untrack(() => projectId);
	const display = new ClassDisplay(project, session.current?.user.id ?? "");
	const [, nz, ny, nx] = manifest.shape_czyx;
	const viewer = new ViewerState([nz, ny, nx], aspectOf(manifest.voxel_size_zyx));
	viewer.window = [...manifest.window];
	viewer.position = [nz / 2, ny / 2, nx / 2];

	let levels: Level[] = $state([]);
	let images: ChunkStore | null = $state(null);
	let labels: LabelLayer | null = $state(null);
	let prediction: Prediction | null = $state.raw(null);
	let live: LivePreview | null = $state.raw(null);
	let modelPlugins: ModelPlugin[] = $state([]);
	let brushFocus: Vec3 | null = $state.raw(null);
	// Your proposal (one ROI predicted on demand), if asked for after the
	// prediction, shown in its box in place of the prediction there.
	let proposal: Prediction | null = $state.raw(null);
	// The model "Propose here" uses: the newest ready one.
	let proposer: { id: string; name: string } | null = $state(null);
	let proposing = $state(false);
	// How far the proposal being made has got (0 to 1), once a worker has it.
	let proposalProgress: number | null = $state(null);
	// The proposal's pipeline, once known, so it can be cancelled.
	let proposalPipeline: string | null = $state(null);
	let cancelling = $state(false);
	let classes: LabelClass[] = $state([]);
	// The classes plus Background, which the person picks from and paints with.
	const pickable = $derived(withBackground(classes, display.backgroundColor));
	// The label values the project has, which the modes that go by classes choose among.
	const labelValues = $derived(pickable.map((c) => c.value));
	// The classes those modes go by now: the ones chosen that the project has, else the active class.
	const paintSet = $derived(viewer.classesFor("paint", labelValues));
	const eraseSet = $derived(viewer.classesFor("erase", labelValues));
	const acceptSet = $derived(viewer.classesFor("accept", labelValues));
	// Whether any background has been painted (by anyone, or here); assumed until the server says not, so the hint doesn't flash.
	let hasBackground = $state(true);
	let paintedBackground = false;
	let error = $state("");
	// Why the prediction layer couldn't load when the page opened; cleared
	// when a later load works.
	let predictionError = $state("");
	// The project's image was replaced since this page opened, so its
	// predictions don't fit the image here.
	let imageReplaced = $state(false);
	let pool: WorkerPool | undefined;
	let hovered: Plane = PLANES.xy;
	// In a four-view layout, the view the pointer is over now, and the one last used (pressed in,
	// scrolled, focused, or keys pressed over): which "this view" is when accepting. Moving to the
	// panel crosses other views, so what the pointer passed over on the way doesn't count.
	let pointed = $state.raw<Plane | null>(null);
	let used = $state.raw<Plane>(PLANES.xy);
	let notice = $state("");
	let classesOpen = $state(true);
	// The form for a new class: open on request, and from the start while the project has none.
	let addingClass = $state(false);
	let className = $state("");
	// The color the person picked, if they did; until then the form offers the next one not in use.
	let pickedColor = $state("");
	const classColor = $derived(pickedColor || nextColor(classes.map((c) => c.color)));
	let classError = $state("");
	let savingClass = $state(false);
	let classInput: HTMLInputElement | null = $state(null);
	let addClassButton: HTMLButtonElement | null = $state(null);
	// Said to screen readers when a class is added.
	let classAdded = $state("");
	const classFormOpen = $derived(addingClass || (labels !== null && classes.length === 0));
	// On narrow screens the dock floats over the views until closed.
	let dockOpen = $state(false);
	const me = session.current?.user.id ?? "";
	const queue = new OpQueue(project, indexedDbStorage(me, project));
	// Strict edits compare against the chunk versions current when they're
	// sent, after this page's earlier edits have landed; if any version is
	// unknown, the edit applies like a brush stroke instead.
	queue.beforeSend = (op: QueuedEdit) => (labels ? strictOn(labels, op) : op);
	const rois = new RoiList(project);
	const firstRoi = untrack(() => startRoi);
	const firstBox = untrack(() => startBox);
	// Each view's size in pixels, which what accepting in a view covers follows.
	const sizes = new SvelteMap<string, [number, number]>();
	const controller = new AbortController();

	const voxelSize = manifest.voxel_size_zyx;
	const unit = manifest.unit ?? "";

	onMount(async () => {
		try {
			levels = await loadLevels(zarrUrl, controller.signal);
			// Opening on a box works without ROIs; opening on an ROI needs them shown.
			(SHOW_ROIS ? rois.load() : Promise.resolve()).then(() => {
				const found = SHOW_ROIS ? rois.items.find((r) => r.id === firstRoi) : undefined;
				if (found) goTo(found);
				else if (firstBox) showBox(firstBox);
				else if (firstRoi) {
					if (SHOW_ROIS) rois.error = "That ROI isn't in this project any more.";
					else notice = "That link points at an ROI, and ROIs are hidden for now.";
				}
			});
			pool = new WorkerPool();
			live = new LivePreview(project, imageId, viewer.shape, pool, me,
				() => accepting || declining || erasingComposite || !!pendingAccept || !!acceptGesture || queue.toggling);
			api<ModelPlugin[]>("/api/plugins", { signal: controller.signal }).then((plugins) => {
				modelPlugins = plugins;
				const first = plugins.find((plugin) => plugin.name === "rf") ?? plugins[0];
				if (first && !controller.signal.aborted) void live?.select(first);
			}, () => {});
			images = new ChunkStore(imageLoader(pool, absolute(zarrUrl), levels), CACHE_BYTES);
			loadPrediction(controller.signal).catch((e: unknown) => {
				if (!controller.signal.aborted) predictionError = message(e);
			});
			// Your proposal still being made (asked for before a reload, say) shows when it's done.
			api<Pipeline[]>(`/api/projects/${project}/pipelines`, { signal: controller.signal }).then(
				(pipelines) => {
					const latest = pipelines.find((p) => p.kind === "proposal" && p.created_by === me);
					if (latest && unfinished(latest) && !proposing) void follow(latest.id);
				},
				() => {},
			);
			const layer = new LabelLayer(project, pool, viewer.shape);
			layer.onRevision = (seq) => live?.changed(seq);
			await layer.start();
			if (controller.signal.aborted) return layer.stop();
			layer.onStopped = () => (error = "Live label updates stopped. Reload the page to see others' edits.");
			labels = layer;
			classes = layer.classes;
			// Classes added here or elsewhere reach the list, and the palette.
			layer.onClasses(() => {
				classes = layer.classes;
				viewer.activeClass ??= classes[0]?.value ?? null;
			});
			viewer.activeClass ??= classes[0]?.value ?? null;
			layer.counts().then(
				(counts) => (hasBackground = paintedBackground || (counts.get(BACKGROUND_VALUE) ?? 0) > 0),
				() => {},
			);
			queue.onOutcome((outcome) => {
				if ("result" in outcome) live?.changed(outcome.result.seq);
				if ("cancelled" in outcome) {
					layer.settle(outcome.op.local, null);
				} else if (outcome.op.kind !== "edit") {
					if ("result" in outcome) layer.changed(outcome.result.chunks);
					else if (!outcome.alreadyDone) notice = `That ${outcome.op.kind} didn't go through: ${outcome.error}`;
				} else if ("result" in outcome) {
					layer.settle(outcome.op.local, outcome.result.chunks);
				} else {
					layer.settle(outcome.op.local, null);
					notice = outcome.conflict
						? "Someone changed those labels while you drew; your polygon was dropped. Draw it again."
						: `That edit didn't save: ${outcome.error}`;
				}
			});
			queue.onRequeue((ops) => {
				for (const op of ops) layer.applyLocal(op.local, op.deltas);
			});
			// Edits a previous visit left unsent show until they're saved.
			for (const op of await queue.start()) if (op.kind === "edit") layer.applyLocal(op.local, op.deltas);
		} catch (e) {
			if (!controller.signal.aborted) error = e instanceof Error ? e.message : String(e);
		}
	});

	const stopRefreshing = SHOW_ROIS ? rois.keepFresh() : () => {};
	// Models trained or deleted, and predictions and proposals made, since.
	// A reload that fails keeps what's shown, quietly, and the next one tries again.
	const stopReloading = whileVisible(() => {
		void loadPrediction(controller.signal).catch(() => {});
		void labels?.refreshClasses().catch(() => {});
	});

	onDestroy(() => {
		controller.abort();
		live?.stop();
		stopRefreshing();
		stopReloading();
		queue.stop();
		labels?.stop();
		images?.keepOnly(new Set());
		pool?.close();
	});

	$effect(() => {
		void [
			viewer.opacity,
			viewer.layout,
			viewer.brushRadius,
			viewer.paintMode,
			viewer.paintClasses,
			viewer.eraseMode,
			viewer.eraseClasses,
			viewer.acceptMode,
			viewer.acceptClasses,
			viewer.roiDepth,
			viewer.showPrediction,
			viewer.predictionOpacity,
			viewer.showSegmentation,
			viewer.segmentationOpacity,
		];
		viewer.savePreferences();
	});

	const shown = $derived(
		viewer.layout === "four" ? [PLANES.xy, PLANES.yz, PLANES.xz] : [PLANES[viewer.layout]],
	);

	$effect(() => {
		if (!live) return;
		live.paused = !viewer.showPrediction;
		const views = shown.flatMap((plane) => {
			const size = sizes.get(plane.name);
			const full = levels[0];
			if (!size || !full) return [];
			const { box } = viewExtent({ plane, position: viewer.position, zoom: viewer.zoom, aspect: viewer.aspect, width: size[0], height: size[1] }, full);
			return box ? [box] : [];
		});
		live.wanted = liveKeys(viewer.shape, viewer.position, views, brushFocus);
	});

	function toggleLive() {
		if (!live) return;
		live.enabled = !live.enabled;
		viewer.showPrediction = live.enabled;
		if (live.enabled && live.plugin) void live.select(live.plugin);
	}

	function fit() {
		viewer.autoFit = true;
		const plane = viewer.layout === "four" ? PLANES.xy : PLANES[viewer.layout];
		const size = sizes.get(plane.name);
		if (!size) return;
		const { shape, aspect } = viewer;
		viewer.zoom = Math.min(size[0] / (shape[plane.u] * aspect[plane.u]), size[1] / (shape[plane.v] * aspect[plane.v]));
		const point = [...viewer.position] as Vec3;
		point[plane.u] = shape[plane.u] / 2;
		point[plane.v] = shape[plane.v] / 2;
		viewer.moveTo(point);
	}

	function resized(plane: Plane, width: number, height: number) {
		sizes.set(plane.name, [width, height]);
		if (pendingPlace) return fitTo(pendingPlace);
		const main = viewer.layout === "four" ? "xy" : viewer.layout;
		if (plane.name === main && viewer.autoFit) fit();
	}

	// --- editing -------------------------------------------------------------

	/** Send one edit of a plane mask, showing it at once. */
	function commit(plane: Plane, slice: number, mask: PlaneMask, value: number, onlyIf: string, tool: Record<string, unknown>, strict = false) {
		if (!labels) return;
		const volume = mask.toVolume(plane, slice);
		const deltas = splitIntoDeltas(volume.mask, volume.shape, volume.origin, { value, onlyIf });
		notice = "";
		for (const op of queue.edit(deltas, { strict, tool })) labels.applyLocal(op.local, op.deltas);
		if (value > 0 && deltas.length > 0 && display.reveal(value)) {
			notice = "Showing the class you just painted.";
		}
		if (value === BACKGROUND_VALUE && deltas.length > 0) hasBackground = paintedBackground = true;
		viewer.revealLabels();
	}

	let erasingComposite = $state(false);

	async function stroke(drawn: Stroke) {
		if (drawn.accept !== undefined) {
			return acceptMask(drawn.accept, drawn.mask, { name: "accept-brush", radius: drawn.radius });
		}
		const focus = [...viewer.position] as Vec3;
		focus[drawn.plane.normal] = drawn.slice;
		focus[drawn.plane.u] = drawn.mask.u0 + drawn.mask.width / 2;
		focus[drawn.plane.v] = drawn.mask.v0 + drawn.mask.height / 2;
		brushFocus = focus;
		if (!drawn.erase && drawn.value === 0) return;
		const tool = {
			name: drawn.erase ? "eraser" : "brush",
			radius: drawn.radius,
			plane: drawn.plane.name,
			slice: drawn.slice,
		};
		if (!drawn.erase) return commit(drawn.plane, drawn.slice, drawn.mask, drawn.value, drawn.onlyIf, tool);
		if (!labels || erasingComposite) {
			if (erasingComposite) notice = "Wait for the previous erase to finish.";
			return;
		}
		const volume = drawn.mask.toVolume(drawn.plane, drawn.slice);
		const box: Box = [
			volume.origin[0],
			volume.origin[1],
			volume.origin[2],
			volume.origin[0] + volume.shape[0],
			volume.origin[1] + volume.shape[1],
			volume.origin[2] + volume.shape[2],
		];
		const layer = showing(box);
		const liveStore = live?.enabled && viewer.showPrediction ? live.store : null;
		const liveRegions = liveStore && live ? [...live.regions.entries()] : [];
		// `onlyIf` was captured when the stroke began, so changing the control
		// while dragging cannot change what the finished gesture does.
		if (!layer && !liveStore && drawn.onlyIf === "unlabeled") return;
		const styles = display.styles;
		const hiddenClasses = labelValues.some((value) => classOpacity(value, styles) === 0);
		if (!layer && !liveStore && drawn.onlyIf !== "labeled" && !hiddenClasses) return commit(drawn.plane, drawn.slice, drawn.mask, 0, drawn.onlyIf, tool);
		if (layer && mixesProposal(box)) {
			notice = "That stroke crosses the edge of a proposal; erase on one side at a time.";
			return;
		}
		if (queue.toggling) {
			notice = "Wait for undo or redo to finish before erasing suggestions.";
			return;
		}
		erasingComposite = true;
		notice = "";
		try {
			const [predictedValues, labeledValues] = await Promise.all([
				liveStore || layer ? readForDecision(liveStore ?? layer!.store, box) : Promise.resolve(new Uint8Array(volume.mask.length)),
				readForDecision(labels.store, box),
			]);
			const declineValues = declinesUnderErase(displayedValues(predictedValues, styles), labeledValues, volume.mask, drawn.onlyIf);
			const decline = declineParts(declineValues, box);
			const chosen = drawn.onlyIf.startsWith("class:") ? new Set(drawn.onlyIf.slice(6).split(",").map(Number)) : null;
			const eraseMask = Uint8Array.from(volume.mask, (selected, index) => {
				const value = labeledValues[index]!;
				return selected &&
					value > 0 &&
					value < DECLINED_VALUE &&
					classOpacity(value, styles) > 0 &&
					(drawn.onlyIf === "any" || drawn.onlyIf === "labeled" || !!chosen?.has(value))
					? 1
					: 0;
			});
			const erases = splitIntoDeltas(eraseMask, volume.shape, volume.origin, { value: 0, onlyIf: "any" }).map((delta) => {
				const version = labels?.versionOf(delta.key.join("/"));
				if (version === undefined) throw new Error("A label chunk changed while the erase was being prepared; try again.");
				return { ...delta, base_version: version };
			});
			const ops = queue.editTogether([
				{ parts: [erases], options: { strict: true, strictPrepared: true, tool } },
				...(liveStore ? liveRegions.flatMap(([id, region]) => {
					const part = intersection(box, region.box);
					return part ? [{ parts: decline.map((deltas) => deltas.filter((delta) => delta.key.join("/") === id)),
						options: { decline: { prediction: region.artifact_id, box: part } } }] : [];
				}) : layer ? [{ parts: decline, options: { decline: { prediction: layer.artifact_id, box } } }] : []),
			]);
			for (const op of ops) labels.applyLocal(op.local, op.deltas);
			if (ops.length > 0) viewer.revealLabels();
		} catch (e) {
			notice = `Couldn't erase the segmentation: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			erasingComposite = false;
		}
	}

	/** Fill the polygon being drawn, or with `cut`, clear the active class inside it. */
	function closePolygon(cut: boolean) {
		const polygon = viewer.polygon;
		viewer.polygon = null;
		if (!polygon) return;
		const plane = PLANES[polygon.plane];
		const limits: [number, number] = [viewer.shape[plane.u], viewer.shape[plane.v]];
		if (polygon.accept !== undefined) {
			const mask = new PlaneMask(...limits);
			mask.polygon(polygon.points);
			void acceptMask(polygon.accept, mask, { name: "accept-polygon" });
			return;
		}
		if (viewer.activeClass === null) return;
		const edit = polygonEdit(polygon, cut ? "subtract" : "add", viewer.activeClass, viewer.paintCondition(labelValues), limits);
		if (edit) commit(plane, polygon.slice, edit.mask, edit.value, edit.onlyIf, edit.tool, true);
	}

	// --- ROIs ----------------------------------------------------------------

	async function drawRoi(plane: Plane, slice: number, corners: [[number, number], [number, number]]) {
		const bbox = roiBox(plane, slice, corners, viewer.roiDepth, viewer.shape);
		if (!bbox) return;
		const roi = await rois.add(bbox, viewer.roiDepth === 1 ? "slice" : "cube");
		if (roi) viewer.selectedRoi = roi.id;
	}

	/** A box to center the views on and fit: an ROI's, or one a link gave. */
	interface Place {
		bbox: Box;
		/** One voxel thick, so it shows in the view of its plane. */
		slice: boolean;
		/** Fit at least this many voxels across, for some room around small boxes. */
		least?: number;
	}

	// A place to fit once the main view knows its size.
	let pendingPlace: Place | null = null;

	/** Select an ROI, and center the views on it and fit it. */
	function goTo(roi: Roi) {
		viewer.selectedRoi = roi.id;
		fitTo({ bbox: roi.bbox, slice: roi.kind === "slice" });
	}

	/** Show a box a link gave, such as an edit's from the history page. */
	function showBox(box: Box) {
		const bbox = clipBox(box, viewer.shape);
		if (!bbox) {
			notice = "That place is outside the image.";
			return;
		}
		fitTo({ bbox, slice: [0, 1, 2].some((a) => bbox[a + 3]! - bbox[a]! === 1), least: 64 });
	}

	/** Center the views on a place and fit it: a slice in the view of its plane, a cube in every view shown. */
	function fitTo(place: Place) {
		viewer.autoFit = false;
		const { bbox } = place;
		const thin = thinAxis(bbox);
		const slicePlane = thin === 0 ? PLANES.xy : thin === 1 ? PLANES.xz : PLANES.yz;
		viewer.moveTo([0, 1, 2].map((a) => (bbox[a]! + bbox[a + 3]!) / 2) as Vec3);
		if (place.slice && viewer.layout !== "four" && viewer.layout !== slicePlane.name) {
			// The new view's size arrives when it lays out; fit then.
			viewer.layout = slicePlane.name;
			sizes.clear();
			pendingPlace = place;
			return;
		}
		const planes = place.slice ? [slicePlane] : shown;
		const extent = (axis: number) => Math.max(place.least ?? 1, bbox[axis + 3]! - bbox[axis]!) * viewer.aspect[axis]!;
		const zooms = planes.flatMap((plane) => {
			const size = sizes.get(plane.name);
			return size ? [Math.min(size[0] / extent(plane.u), size[1] / extent(plane.v))] : [];
		});
		pendingPlace = zooms.length === planes.length ? null : place;
		if (zooms.length > 0) viewer.zoom = Math.min(64, Math.max(1 / 512, 0.85 * Math.min(...zooms)));
	}

	/** A prediction, as the server describes it; a proposal's also has its box. */
	interface Predicted {
		artifact_id: string;
		zarr_url: string;
		model_id: string | null;
		model_name: string | null;
		image_artifact_id: string;
		shape_zyx: number[];
		// When it was asked for, and when it was done.
		started_at: string;
		committed_at: string;
		box?: Box;
	}

	/**
	 * A prediction the prediction layer shows: of the whole image, or a
	 * proposal, which holds values only inside its box.
	 */
	interface Prediction extends Predicted {
		kind: "prediction" | "proposal";
		box: Box;
		store: ChunkStore;
	}

	// Each load of the prediction layer, so only the latest one's answers count.
	let loads = 0;

	/**
	 * Load the prediction, and your proposal if it was asked for after it, for
	 * the prediction layer; and find the model "Propose here" uses. Throws if
	 * the server couldn't say, leaving the layer as it was.
	 */
	async function loadPrediction(signal: AbortSignal) {
		if (imageReplaced || acceptGesture || accepting) return;
		const load = ++loads;
		const get = (slot: string) =>
			api<Predicted>(`/api/projects/${project}/${slot}`, { signal }).catch((e: unknown) => {
				if (e instanceof ApiError && e.status === 404) return null;
				throw e;
			});
		const [whole, proposed, models] = await Promise.all([
			get("prediction"),
			get("proposal"),
			// Without them, "Propose here" keeps the model it had.
			api<{ id: string; name: string; status: string }[]>(`/api/projects/${project}/models`, { signal }).catch(
				() => null,
			),
		]);
		if (signal.aborted || load !== loads || imageReplaced || acceptGesture || accepting) return;
		// The server gives only predictions of the project's current image, so
		// one of another image means that image replaced the one shown here.
		if ([whole, proposed].some((found) => found && found.image_artifact_id !== imageId)) {
			imageReplaced = true;
			prediction = proposal = null;
			predictionError = "This project's image was replaced; reload the page to see the new one.";
			return;
		}
		predictionError = "";
		// Newest first.
		if (models) proposer = models.find((m) => m.status === "ready") ?? null;
		const image: Box = [0, 0, 0, ...viewer.shape];
		prediction = show(prediction, whole && { ...whole, box: image }, "prediction");
		// A proposal asked for before the prediction is out of date; one asked
		// for after it shows, even if the prediction was done later.
		proposal = show(
			proposal,
			SHOW_ROIS && proposed?.box && (!whole || Date.parse(proposed.started_at) > Date.parse(whole.started_at)) ? proposed : null,
			"proposal",
		);
	}

	/** `found` as the layer shows it, with `before`'s chunks if it's the same one. */
	function show(before: Prediction | null, found: Predicted | null, kind: Prediction["kind"]): Prediction | null {
		if (!found?.box || !pool) return null;
		if (before?.artifact_id === found.artifact_id) return before;
		// Predictions never change once made, so their chunks cache like the image's.
		const store = new ChunkStore(labelLoader(pool, absolute(found.zarr_url), viewer.shape), 128 * 1024 * 1024, 4);
		return { ...found, box: found.box, kind, store };
	}

	/** The part of an ROI inside the image, or null if none is. */
	function inImage(roi: Roi): Box | null {
		return clipBox(roi.bbox, viewer.shape);
	}

	/**
	 * The layer a box (inside the image) shows, which accepting there reads:
	 * the proposal if the box is inside its box, else the prediction.
	 */
	function showing(box: Box): Prediction | null {
		if (!viewer.showPrediction) return null;
		if (live?.enabled) {
			const region = [...live.regions.values()].find((r) => within(box, r.box));
			return region ? { ...region, kind: "prediction",
				image_artifact_id: imageId, shape_zyx: viewer.shape,
				started_at: "", committed_at: "" } : null;
		}
		return [proposal, prediction].find((layer) => layer && within(box, layer.box)) ?? null;
	}

	/** The layer an ROI shows, which accepting there reads. */
	function covering(roi: Roi): Prediction | null {
		const box = inImage(roi);
		return box ? showing(box) : null;
	}

	/**
	 * Whether a box reaches past the proposal's box, so part of it shows the
	 * proposal and part the prediction, and accepting reads only one of them
	 * (unless the same model made both, so they agree).
	 */
	function mixesProposal(box: Box): boolean {
		if (live?.enabled) return false;
		if (!proposal || !overlaps(box, proposal.box) || within(box, proposal.box)) return false;
		return !(prediction && prediction.model_id === proposal.model_id);
	}

	/** Why accepting in an ROI can't go ahead because of that, if it can't. */
	function acceptBlockedIn(roi: Roi): string {
		const box = inImage(roi);
		return box && mixesProposal(box) ? "Part of this ROI shows your proposal and part doesn't; propose the whole ROI to accept it" : "";
	}

	/**
	 * What accepting in the view would do: the part of its slice the active
	 * view shows (once the view has a size), and how many chunks that view
	 * counts to draw the prediction there (see `viewExtent`, which counts them
	 * as the view does, so accepting is on only where the prediction is
	 * drawn), the layer that shows there, which accepting reads, and why it
	 * can't go ahead, if it can't. These words are the button's tooltip and the
	 * notice for the key.
	 */
	const inView = $derived.by(() => {
		const plane = planeToAccept(viewer.layout, pointed, used);
		const size = sizes.get(plane.name);
		const full = levels[0];
		const { box, tiles } =
			size && full
				? viewExtent({ plane, position: viewer.position, zoom: viewer.zoom, aspect: viewer.aspect, width: size[0], height: size[1] }, full)
				: { box: null, tiles: 0 };
		const layer = box ? showing(box) : null;
		const blocked = whyNotInView({
			imageReplaced,
			predicted: live?.enabled ? !!live.regions.size : !!(prediction || proposal),
			shown: viewer.showPrediction,
			opacity: viewer.opacity,
			box,
			tiles,
			mixed: !!box && mixesProposal(box),
			covered: live?.enabled ? !!box && [...live.regions.values()].some((r) => intersection(box, r.box)) : !!layer,
		});
		return { plane, box, layer, blocked };
	});

	/**
	 * Follow a pipeline, passing on its updates, until it ends or its stream
	 * closes for good (the server refused it, say), and give its state then:
	 * in the second case it may still be unfinished. Stops following, and
	 * rejects, when `signal` aborts.
	 */
	function finished(id: string, signal: AbortSignal, onupdate: (pipeline: Pipeline) => void): Promise<Pipeline> {
		return new Promise((resolve, reject) => {
			if (signal.aborted) return reject(signal.reason);
			const source = new EventSource(`/api/projects/${project}/pipelines/${id}/events`);
			let ended = false;
			/** Stop following; false if that already happened. */
			const stop = () => {
				if (ended) return false;
				ended = true;
				source.close();
				signal.removeEventListener("abort", aborted);
				return true;
			};
			const aborted = () => {
				if (stop()) reject(signal.reason);
			};
			signal.addEventListener("abort", aborted);
			source.addEventListener("status", (event) => {
				const update = JSON.parse((event as MessageEvent<string>).data) as Pipeline;
				if (unfinished(update)) onupdate(update);
				else if (stop()) resolve(update);
			});
			// The server answers 204 for a pipeline that has ended, which closes the stream.
			source.addEventListener("error", () => {
				if (source.readyState !== EventSource.CLOSED || !stop()) return;
				api<Pipeline>(`/api/projects/${project}/pipelines/${id}`, { signal }).then(resolve, reject);
			});
		});
	}

	/** Why a proposal wasn't made, in a sentence. */
	async function whyNot(final: Pipeline): Promise<string> {
		if (unfinished(final)) return "Lost track of the proposal; it shows here once it's made.";
		if (final.status === "failed") return final.error ?? "The proposal failed.";
		// Each of your proposals stops your others still running.
		const pipelines = await api<Pipeline[]>(`/api/projects/${project}/pipelines`, { signal: controller.signal }).catch(
			() => [],
		);
		const newest = pipelines.find((p) => p.kind === "proposal" && p.created_by === me);
		return newest && newest.id !== final.id ? "Your newer proposal replaced this one." : "The proposal was cancelled.";
	}

	/**
	 * Show how far a proposal (its pipeline, or a request that gives it) has
	 * got, and then the proposal.
	 */
	async function follow(pipeline: string | Promise<string>) {
		proposing = true;
		proposalProgress = null;
		try {
			proposalPipeline = await pipeline;
			const final = await finished(proposalPipeline, controller.signal, (update) => {
				proposalProgress = update.status === "running" ? update.progress : null;
			});
			if (final.status !== "succeeded") {
				// Cancelling it was what was asked for, so it needs no notice.
				if (!(cancelling && final.status === "cancelled")) notice = await whyNot(final);
				return;
			}
			await loadPrediction(controller.signal);
			viewer.showPrediction = true;
		} catch (e) {
			if (!controller.signal.aborted) notice = message(e);
		} finally {
			proposing = false;
			proposalPipeline = null;
			cancelling = false;
		}
	}

	/** Stop the proposal being made; it ends as cancelled once its worker stops. */
	async function cancelProposal() {
		if (!proposalPipeline || cancelling) return;
		cancelling = true;
		try {
			await api(`/api/projects/${project}/pipelines/${proposalPipeline}/cancel`, { method: "POST" });
		} catch (e) {
			cancelling = false;
			notice = message(e);
		}
	}

	/** Predict just this ROI with the proposer, and show the result to accept. */
	function propose(roi: Roi) {
		if (!proposer || proposing) return;
		notice = "";
		const started = api<{ pipeline_id: string }>(`/api/projects/${project}/models/${proposer.id}/propose`, {
			body: { roi_id: roi.id },
			signal: controller.signal,
		});
		void follow(started.then((s) => s.pipeline_id));
	}

	const openRois = $derived(rois.items.filter((r) => r.status === "open"));
	const selectedRoi = $derived(rois.items.find((r) => r.id === viewer.selectedRoi) ?? null);
	// What the selected ROI shows, and accepting there reads.
	const shownHere = $derived(selectedRoi ? covering(selectedRoi) : null);
	const acceptBlocked = $derived(selectedRoi ? acceptBlockedIn(selectedRoi) : "");
	// Why proposing for the selected ROI can't help, if it can't.
	const proposeBlocked = $derived.by(() => {
		if (!selectedRoi || !proposer) return "";
		if (imageReplaced) return "This project's image was replaced; reload the page first";
		const box = inImage(selectedRoi);
		if (!box) return "That ROI is outside the image";
		if (voxels(box) > MAX_PROPOSAL_VOXELS) {
			return "Proposals are for ROIs up to 256³ voxels; predict the whole image on the Models page instead";
		}
		if (shownHere?.model_id === proposer.id) return `The ${shownHere.kind} here is already from ${proposer.name}, the newest model`;
		return "";
	});
	const proposeLabel = $derived(
		!proposing ? "Propose here" : proposalProgress === null ? "Proposing…" : `Proposing… ${Math.round(proposalProgress * 100)}%`,
	);
	let accepting = $state(false);
	let declining = $state(false);
	let restoring = $state(false);
	let gestureSequence = 0;
	let acceptedVoxels = $state<number | null>(null);
	let acceptGesture = $state.raw<{
		id: number;
		plane: Plane;
		slice: number;
		sources: GestureSource[];
		proposed: GestureSource | null;
		styles: ClassStyles;
		classes: Set<number>;
	} | null>(null);
	const decidingSuggestion = $derived(accepting || declining || restoring || !!acceptGesture);

	/** Only already-loaded chunks can contribute to an Accept gesture. */
	const readLoaded = (store: ChunkStore, box: Box) => readBox((id) => {
		const chunk = store.peek(id);
		if (!chunk) return Promise.reject(new Error("Some predictions or labels are still loading; wait and try that gesture again."));
		return Promise.resolve(chunk);
	}, box);

	function beginAcceptGesture(plane: Plane, slice: number): number | null {
		if (!labels || decidingSuggestion || erasingComposite || pendingAccept || queue.toggling) {
			notice = "Finish the current gesture or wait for the previous action first.";
			return null;
		}
		const size = sizes.get(plane.name), full = levels[0];
		const extent = size && full ? viewExtent({ plane, position: viewer.position, zoom: viewer.zoom,
			aspect: viewer.aspect, width: size[0], height: size[1] }, full) : { box: null, tiles: 0 };
		const available = live?.enabled ? [...live.regions.values()] : [proposal, prediction].filter((p) => p !== null);
		const blocked = whyNotInView({
			imageReplaced, predicted: available.length > 0, shown: viewer.showPrediction, opacity: viewer.opacity,
			...extent, mixed: false, covered: !!extent.box && available.some((p) => intersection(p.box, extent.box!)),
		});
		if (blocked) { notice = blocked; return null; }
		const source = (p: { artifact_id: string; box: Box; store: ChunkStore }): GestureSource | null => {
			// Pointer capture can carry a brush outside the canvas. Cached data
			// beyond its visible slice must not become accepted unseen.
			const box = intersection(p.box, extent.box!);
			return box ? { artifact: p.artifact_id, box, read: (part) => readLoaded(p.store, part) } : null;
		};
		acceptGesture = {
			id: ++gestureSequence, plane, slice,
			sources: (live?.enabled ? [...live.regions.values()] : prediction ? [prediction] : [])
				.map(source).filter((p) => p !== null),
			proposed: !live?.enabled && proposal ? source(proposal) : null,
			styles: display.styles, classes: new Set(viewer.acceptValues(labelValues)),
		};
		notice = "";
		acceptedVoxels = null;
		return acceptGesture.id;
	}

	function cancelAcceptGesture(id?: number) {
		if (id !== undefined && acceptGesture?.id !== id) return;
		acceptGesture = null;
		if (viewer.polygon?.accept !== undefined) viewer.polygon = null;
	}

	async function acceptMask(id: number, mask: PlaneMask, tool: Record<string, unknown>) {
		const frozen = acceptGesture;
		if (!frozen || frozen.id !== id || !labels || accepting) return;
		accepting = true;
		try {
			if (!viewer.showPrediction || viewer.opacity <= 0 || imageReplaced || queue.toggling) {
				throw new Error("The prediction is no longer visible or labels are changing; try again.");
			}
			const box = gestureBox(frozen.plane, frozen.slice, mask);
			let sources = frozen.sources;
			if (mask.count && frozen.proposed && overlaps(box, frozen.proposed.box)) {
				if (!within(box, frozen.proposed.box)) throw new Error("That gesture crosses the proposal edge; accept on one side at a time.");
				sources = [frozen.proposed];
			}
			const prepared = await acceptGestureGroups(frozen.plane, frozen.slice, mask, sources,
				(box) => readLoaded(labels!.store, box), frozen.styles, frozen.classes,
				{ ...tool, plane: frozen.plane.name, slice: frozen.slice });
			// Esc, a tool switch, or leaving the page can cancel the asynchronous read.
			if (acceptGesture?.id !== id || controller.signal.aborted) return;
			if (!viewer.showPrediction || viewer.opacity <= 0 || imageReplaced || queue.toggling) {
				throw new Error("The prediction is no longer visible or labels are changing; try again.");
			}
			for (const op of queue.editTogether(prepared.groups)) labels.applyLocal(op.local, op.deltas);
			acceptedVoxels = prepared.count;
			if (prepared.count) viewer.revealLabels();
		} catch (e) {
			if (acceptGesture?.id === id && !controller.signal.aborted) notice = message(e);
		} finally {
			cancelAcceptGesture(id);
			accepting = false;
		}
	}
	type AcceptWhere = { roi: string } | { box: Box };
	interface PendingAccept {
		pieces?: { artifact: string; box: Box; values: Uint8Array }[];
		artifact: string;
		box: Box;
		where: AcceptWhere;
		values: Uint8Array;
		kind: Prediction["kind"];
		place: string;
		classes: number;
		background: number;
	}
	// Acceptance is prepared before it is sent so Background is a visible,
	// counted choice rather than an accidental part of the default action.
	let pendingAccept: PendingAccept | null = $state.raw(null);
	let includeAcceptBackground = $state(false);

	/** Read a box without tying its requests to a view's cancellable wants. */
	const readForDecision = (from: ChunkStore, box: Box) =>
		readBox((id) => {
			from.want(`decision:${id}`, new Set([id]));
			return (from.loading(id) ?? from.request(id)).finally(() => from.want(`decision:${id}`, new Set()));
		}, box);

	/**
	 * Copy the model's prediction inside an ROI into the labels, as accepted
	 * (model-verified) labels, without touching voxels anyone labeled. It
	 * reads what the ROI shows: the proposal, if the ROI is inside its box,
	 * else the prediction.
	 */
	async function acceptPrediction(roi: Roi) {
		if (!labels || decidingSuggestion) return;
		// Only the part of the ROI inside the image has anything to accept.
		const box = inImage(roi);
		if (!box) {
			notice = "That ROI is outside the image.";
			return;
		}
		const blocked = acceptBlockedIn(roi);
		if (blocked) {
			notice = `${blocked}.`;
			return;
		}
		// Kept whole, so a newer prediction or proposal arriving while this
		// reads doesn't change what it accepts from.
		const layer = covering(roi);
		if (!layer) {
			if (proposal) notice = "Nothing is predicted in that ROI yet; propose it first.";
			return;
		}
		if (voxels(box) > MAX_ACCEPT_VOXELS) {
			notice = "That ROI is too big to accept at once; draw a smaller one.";
			return;
		}
		await prepareAccept(layer, box, { roi: roi.id });
	}

	/**
	 * Copy what the active view shows of the model's prediction (the part of
	 * its slice the view covers, see `inView`) into the labels, as accepted
	 * (model-verified) labels, without touching voxels anyone labeled. Undo
	 * takes it back in one step. If it can't go ahead, the notice says why.
	 */
	async function acceptView() {
		if (!labels || decidingSuggestion) return;
		const { plane, box, layer, blocked } = inView;
		const place = `the ${reviewSlice(plane, Math.floor(viewer.position[plane.normal]))}`;
		if (blocked) notice = blocked;
		else if (box && live?.enabled) await prepareLive(box, place);
		else if (box && layer) await prepareAccept(layer, box, { box }, place);
	}

	async function prepareLive(box: Box, place: string) {
		if (!live || !labels || queue.toggling || accepting || pendingAccept) return;
		accepting = true;
		notice = "";
		const regions = [...live.regions.values()];
		try {
			const pieces = [];
			let classes = 0, background = 0;
			for (const region of regions) {
				const part = intersection(box, region.box);
				if (!part) continue;
				const values = displayedValues(unlabeledOnly(await readForDecision(region.store, part), await readForDecision(labels.store, part)), display.styles);
				const counts = acceptCounts(values);
				classes += counts.classes;
				background += counts.background;
				pieces.push({ artifact: region.artifact_id, box: part, values });
			}
			if (classes + background === 0) { notice = "No unannotated suggestions from visible classes in the ready part of this view."; return; }
			includeAcceptBackground = false;
			pendingAccept = { artifact: "", box, where: { box }, values: new Uint8Array(), pieces,
				kind: "prediction", place, classes, background };
		} catch (e) { notice = message(e); }
		finally { accepting = false; }
	}

	/**
	 * Read the prediction and current labels in `box`, then stage exactly the
	 * still-unlabeled suggestions for confirmation. This makes the Background
	 * choice and its count explicit before anything is written.
	 */
	async function prepareAccept(layer: Prediction, box: Box, where: AcceptWhere, place = "that ROI") {
		if (!labels) return;
		if (queue.toggling) {
			notice = "Wait for undo or redo to finish before accepting a prediction.";
			return;
		}
		accepting = true;
		notice = "";
		const { store, artifact_id: artifact, kind } = layer;
		try {
			const predictedValues = await readForDecision(store, box);
			const values = displayedValues(unlabeledOnly(predictedValues, await readForDecision(labels.store, box)), display.styles);
			const counts = acceptCounts(values);
			const predicted = values.some((value) => value > 0);
			if (counts.classes > 0 || counts.background > 0) {
				includeAcceptBackground = false;
				pendingAccept = { artifact, box, where, values, kind, place, ...counts };
			} else if (!predictedValues.some((value) => value > 0)) notice = `The ${kind} has nothing in ${place}.`;
			else if (!predicted) notice = `The suggestions in ${place} are already labeled or their classes are hidden.`;
		} catch (e) {
			notice = `Couldn't prepare the ${kind}: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			accepting = false;
		}
	}

	/** Send the staged foreground, plus Background only when explicitly selected. */
	function confirmAccept() {
		if (!labels || !pendingAccept) return;
		const pending = pendingAccept;
		if (pending.classes === 0 && !includeAcceptBackground) return;
		const ops = pending.pieces
			? queue.editTogether(pending.pieces.map((piece) => ({ parts: acceptParts(piece.values, piece.box, includeAcceptBackground), options: { accept: { prediction: piece.artifact, box: piece.box } } })))
			: queue.editMany(acceptParts(pending.values, pending.box, includeAcceptBackground), { accept: { prediction: pending.artifact, ...pending.where } });
		for (const op of ops) labels.applyLocal(op.local, op.deltas);
		if (ops.length > 0) viewer.revealLabels();
		pendingAccept = null;
		includeAcceptBackground = false;
	}

	function cancelAccept() {
		pendingAccept = null;
		includeAcceptBackground = false;
	}

	/** Decline foreground suggestions in an ROI, after authenticating them against their prediction. */
	async function declinePrediction(roi: Roi) {
		if (!labels || decidingSuggestion) return;
		const box = inImage(roi);
		if (!box) {
			notice = "That ROI is outside the image.";
			return;
		}
		const blocked = acceptBlockedIn(roi);
		if (blocked) {
			notice = `${blocked}.`;
			return;
		}
		const layer = covering(roi);
		if (!layer) return;
		if (voxels(box) > MAX_ACCEPT_VOXELS) {
			notice = "That ROI is too big to decline at once; draw a smaller one.";
			return;
		}
		await declineFrom(layer, box, { roi: roi.id });
	}

	/** Decline foreground suggestions in the active view. */
	async function declineView() {
		if (!labels || decidingSuggestion) return;
		const { box, layer, blocked } = inView;
		if (blocked) notice = blocked;
		else if (box && live?.enabled && labels) {
			declining = true;
			try {
				const regions = [...live.regions.values()];
				const groups = [];
				for (const region of regions) {
					const part = intersection(box, region.box);
					if (!part) continue;
					const values = displayedValues(unlabeledOnly(await readForDecision(region.store, part), await readForDecision(labels.store, part)), display.styles);
					groups.push({ parts: declineParts(values, part), options: { decline: { prediction: region.artifact_id, box: part } } });
				}
				for (const op of queue.editTogether(groups)) labels.applyLocal(op.local, op.deltas);
			} catch (e) { notice = message(e); }
			finally { declining = false; }
		}
		else if (box && layer) await declineFrom(layer, box, { box });
	}

	async function declineFrom(layer: Prediction, box: Box, where: AcceptWhere) {
		if (!labels) return;
		if (queue.toggling) {
			notice = "Wait for undo or redo to finish before declining suggestions.";
			return;
		}
		declining = true;
		notice = "";
		const { store, artifact_id: artifact, kind } = layer;
		try {
			const predictedValues = await readForDecision(store, box);
			const values = displayedValues(unlabeledOnly(predictedValues, await readForDecision(labels.store, box)), display.styles);
			const parts = declineParts(values, box);
			const ops = queue.editMany(parts, { decline: { prediction: artifact, ...where } });
			for (const op of ops) labels.applyLocal(op.local, op.deltas);
			const place = "roi" in where ? "that ROI" : "this view";
			if (ops.length === 0) {
				if (!predictedValues.some((value) => value > BACKGROUND_VALUE && value < DECLINED_VALUE)) {
					notice = `The ${kind} has no foreground suggestions in ${place}.`;
				} else notice = `The foreground suggestions in ${place} are already labeled, declined, or their classes are hidden.`;
			}
		} catch (e) {
			notice = `Couldn't decline the ${kind}: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			declining = false;
		}
	}

	/** Restore hidden suggestions in an ROI by removing only its decline tombstones. */
	async function restorePrediction(roi: Roi) {
		if (!labels || decidingSuggestion) return;
		const box = inImage(roi);
		if (!box) {
			notice = "That ROI is outside the image.";
			return;
		}
		if (voxels(box) > MAX_ACCEPT_VOXELS) {
			notice = "That ROI is too big to restore at once; draw a smaller one.";
			return;
		}
		await restoreDeclined(box, "that ROI");
	}

	/** Restore hidden suggestions in the active view. */
	async function restoreView() {
		if (!labels || decidingSuggestion) return;
		const { box, blocked } = inView;
		if (blocked) notice = blocked;
		else if (box) await restoreDeclined(box, "this view");
	}

	async function restoreDeclined(box: Box, place: string) {
		if (!labels) return;
		if (queue.toggling) {
			notice = "Wait for undo or redo to finish before restoring suggestions.";
			return;
		}
		restoring = true;
		notice = "";
		try {
			const parts = restoreDeclinedParts(await readForDecision(labels.store, box), box);
			const ops = queue.editMany(parts, { tool: { name: "restore-declined" } });
			for (const op of ops) labels.applyLocal(op.local, op.deltas);
			if (ops.length === 0) notice = `There are no declined suggestions in ${place}.`;
		} catch (e) {
			notice = `Couldn't restore suggestions: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			restoring = false;
		}
	}

	let exploring = $state(false);

	/**
	 * Make an ROI at a random place no ROI covers, go there, and propose
	 * labels there with the newest model, if one is ready.
	 */
	async function explore() {
		if (exploring) return;
		exploring = true;
		notice = "";
		try {
			const roi = await rois.explore();
			if (!roi) return;
			goTo(roi);
			if (proposer && !proposing && !proposeBlocked) propose(roi);
		} finally {
			exploring = false;
		}
	}

	/** The next open ROI after the selected one, in list order. */
	function nextOpen() {
		const items = rois.items;
		const at = items.findIndex((r) => r.id === viewer.selectedRoi);
		for (let step = 1; step <= items.length; step++) {
			const roi = items[(at + step) % items.length]!;
			if (roi.status === "open") return goTo(roi);
		}
	}

	function setTool(tool: typeof viewer.tool) {
		if (viewer.tool !== tool) {
			cancelAcceptGesture();
			viewer.polygon = null;
			acceptedVoxels = null;
		}
		viewer.tool = tool;
		if (!viewer.drawingPolygon) viewer.polygon = null;
	}

	function setAcceptShape(shape: typeof viewer.acceptShape) {
		if (shape === viewer.acceptShape) return;
		cancelAcceptGesture();
		viewer.polygon = null;
		viewer.acceptShape = shape;
		acceptedVoxels = null;
	}

	function historyStep(redo = false) {
		cancelAcceptGesture();
		acceptedVoxels = null;
		if (redo) void queue.redo();
		else void queue.undo();
	}

	// Going by classes, the ones the menu shows are the ones used from then on,
	// not whichever the active class is when it changes.
	function setPaintMode(mode: PaintMode) {
		if (mode === "classes") viewer.paintClasses = paintSet;
		viewer.paintMode = mode;
	}

	function setEraseMode(mode: EraseMode) {
		if (mode === "classes") viewer.eraseClasses = eraseSet;
		viewer.eraseMode = mode;
	}

	function setAcceptMode(mode: AcceptMode) {
		if (mode === viewer.acceptMode) return;
		cancelAcceptGesture();
		acceptedVoxels = null;
		if (mode === "classes") viewer.acceptClasses = acceptSet;
		viewer.acceptMode = mode;
	}

	function setAcceptClasses(values: number[]) {
		cancelAcceptGesture();
		acceptedVoxels = null;
		viewer.acceptClasses = values;
	}

	const status = $derived(
		{
			offline: `Offline · ${queue.pending} waiting`,
			retrying: `Can't save right now · ${queue.pending} waiting`,
			error: queue.error,
			saving: `Saving ${queue.pending}`,
			saved: "Saved",
		}[saveState(queue)],
	);

	function keyUp(event: KeyboardEvent) {
		viewer.noteKeys(event);
		holdAlt(event);
		if (event.key === " ") viewer.panning = false;
	}

	/** Alt held to cut out a polygon shouldn't open the browser's menu, as Alt alone does on Windows and Linux. */
	function holdAlt(event: KeyboardEvent) {
		if (event.key === "Alt" && viewer.tool === "polygon" && !forFocused(event)) event.preventDefault();
	}

	/** Leaving the window lets go of every key. */
	function blurred() {
		viewer.panning = false;
		viewer.noteKeys({ altKey: false, shiftKey: false });
	}

	// Keys that move the views, which wait while a polygon is dragged out on its slice.
	const MOVES = new Set<Action>(["slice-next", "slice-previous", "fit", "next-roi", "layout"]);

	/** The view that slice keys step: the only one, or the one last pointed at. */
	function activePlane(): Plane {
		return viewer.layout === "four" ? hovered : PLANES[viewer.layout];
	}

	// A view that went away took the pointer with it.
	$effect(() => {
		void viewer.layout;
		pointed = null;
	});

	function key(event: KeyboardEvent) {
		viewer.noteKeys(event);
		holdAlt(event);
		if (pendingAccept && event.key === "Escape") {
			cancelAccept();
			event.preventDefault();
			return;
		}
		if (viewer.help && event.key === "Escape") {
			viewer.help = false;
			event.preventDefault();
			return;
		}
		if (event.key === " " && !forFocused(event)) {
			viewer.panning = true;
			event.preventDefault();
			return;
		}
		const action = actionFor(event);
		if (!action) return;
		if (pointed) used = pointed;
		// Enter and Backspace only mean something while drawing a polygon.
		if ((action === "close-polygon" || action === "remove-point") && !viewer.polygon) return;
		event.preventDefault();
		if (viewer.lassoing && MOVES.has(action)) return;
		if (acceptGesture && (MOVES.has(action) || action === "zoom-in" || action === "zoom-out")) return;
		const step = event.shiftKey ? 10 : 1;
		switch (action) {
			case "navigate":
			case "brush":
			case "eraser":
			case "polygon":
			case "roi":
				return setTool(action);
			case "next-roi":
				return nextOpen();
			case "accept-tool":
				return setTool("accept");
			case "accept": {
				// Holding the key down would go on to accept what's already accepted.
				const target = acceptKeyTarget(event.repeat, selectedRoi !== null);
				if (pendingAccept) {
					if (target !== "nothing") confirmAccept();
					return;
				}
				if (target === "roi" && selectedRoi) void acceptPrediction(selectedRoi);
				else if (target === "view") void acceptView();
				return;
			}
			case "decline": {
				const target = acceptKeyTarget(event.repeat, selectedRoi !== null);
				if (target === "roi" && selectedRoi) void declinePrediction(selectedRoi);
				else if (target === "view") void declineView();
				return;
			}
			case "complete-roi":
				if (viewer.selectedRoi) rois.update(viewer.selectedRoi, { status: event.shiftKey ? "open" : "complete" });
				return;
			case "smaller":
				viewer.brushRadius = Math.max(0.5, Math.round(viewer.brushRadius / 1.25 * 2) / 2);
				return;
			case "bigger":
				viewer.brushRadius = Math.min(64, Math.max(viewer.brushRadius + 0.5, Math.round(viewer.brushRadius * 1.25 * 2) / 2));
				return;
			case "class": {
				const chosen = pickable[Number(event.key) - 1];
				if (chosen) viewer.activeClass = chosen.value;
				return;
			}
			case "close-polygon":
				return closePolygon(closingMode(viewer.polygonMode, event) === "subtract");
			case "remove-point":
				if (viewer.polygon) {
					if (viewer.polygon.points.length <= 1 && viewer.polygon.accept !== undefined) cancelAcceptGesture();
					else viewer.polygon = { ...viewer.polygon, points: viewer.polygon.points.slice(0, -1) };
				}
				return;
			case "cancel":
				if (acceptGesture) cancelAcceptGesture();
				else if (viewer.polygon) viewer.polygon = null;
				else setTool("navigate");
				return;
			case "undo":
				return historyStep();
			case "redo":
				return historyStep(true);
			case "slice-next":
				return viewer.step(activePlane().normal, step);
			case "slice-previous":
				return viewer.step(activePlane().normal, -step);
			case "zoom-in":
				viewer.autoFit = false;
				return viewer.zoomBy(1.25);
			case "zoom-out":
				viewer.autoFit = false;
				return viewer.zoomBy(0.8);
			case "fit":
				return fit();
			case "layout":
				viewer.nextLayout();
				return;
			case "labels":
				viewer.showLabels = !viewer.showLabels;
				return;
			case "prediction":
				cancelAcceptGesture();
				viewer.showPrediction = !viewer.showPrediction;
				return;
			case "help":
				viewer.help = !viewer.help;
				return;
		}
	}

	/**
	 * Clicking a button in here leaves focus where it was, so Space still pans
	 * rather than pressing the button again. Tab still reaches the buttons.
	 */
	function keepFocus(node: HTMLElement) {
		const down = (event: MouseEvent) => {
			if ((event.target as Element).closest("button")) event.preventDefault();
		};
		node.addEventListener("mousedown", down);
		return () => node.removeEventListener("mousedown", down);
	}

	/** Focus the keys dialog while it's open, and give focus back to whatever had it. */
	function holdFocus(dialog: HTMLElement) {
		const opener = document.activeElement;
		dialog.focus();
		return () => {
			if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
		};
	}

	function helpKey(event: KeyboardEvent) {
		// The viewer's keys wait until the dialog closes.
		event.stopPropagation();
		if (event.key === "Escape" || event.key === "?") {
			event.preventDefault();
			viewer.help = false;
		} else if (event.key === "Tab") {
			// Tab cycles through the dialog's controls without leaving it.
			event.preventDefault();
			const dialog = event.currentTarget as HTMLElement;
			const stops = [...dialog.querySelectorAll<HTMLElement>("a[href], button, input, select, textarea")];
			const at = stops.indexOf(document.activeElement as HTMLElement);
			stops.at(event.shiftKey ? (at <= 0 ? -1 : at - 1) : (at + 1) % stops.length)?.focus();
		}
	}

	function voxel(axis: number): string {
		const index = Math.floor(viewer.position[axis]!);
		if (!voxelSize || !unit) return String(index);
		return `${index} (${(index * voxelSize[axis]!).toFixed(1)} ${unit})`;
	}

	const TOOLS = ([
		{ tool: "navigate", label: "Navigate", shortcut: "N", icon: Hand },
		{ tool: "brush", label: "Brush", shortcut: "B", icon: Brush },
		{ tool: "eraser", label: "Eraser", shortcut: "E", icon: Eraser },
		{ tool: "polygon", label: "Polygon", shortcut: "P", icon: Pentagon },
		{ tool: "accept", label: "Accept", shortcut: "I", icon: CheckCheck },
		{ tool: "roi", label: "ROI", shortcut: "R", icon: SquareDashed },
	] as const).filter((entry) => SHOW_ROIS || entry.tool !== "roi");

	const LAYOUT_NAMES = { four: "Four views", xy: "XY", xz: "XZ", yz: "YZ" } as const;
	const ACCEPT_SHAPES = [
		{ value: "brush", label: "Brush", icon: Brush, title: "Brush over visible foreground predictions to accept them" },
		{ value: "polygon", label: "Polygon", icon: Pentagon, title: "Outline visible foreground predictions to accept them" },
	] as const;
	const LAYOUT_OPTIONS = LAYOUTS.map((layout) => ({
		value: layout,
		label: LAYOUT_NAMES[layout],
		icon: layout === "four" ? LayoutGrid : undefined,
	}));

	const POLYGON_MODES: { value: PolygonMode; label: string; icon: typeof SquaresUnite; title: string }[] = [
		{
			value: "add",
			label: "Add",
			icon: SquaresUnite,
			title: "Closing fills the shape with the active class, where the paint mode allows (hold Shift to fill in either mode)",
		},
		{
			value: "subtract",
			label: "Subtract",
			icon: SquaresSubtract,
			title: "Closing cuts the shape out of the active class, leaving other classes: it has its own rule, whatever the paint mode says (hold Alt to cut out in either mode)",
		},
	];

	const activeClass = $derived(pickable.find((c) => c.value === viewer.activeClass));

	function previewClassColor(value: number, color: string) {
		if (value === BACKGROUND_VALUE) display.set(value, { color });
		else labels?.previewColor(value, color);
	}

	/** Open the new-class form, its name ready to type (what's typed already stays). */
	async function startClass() {
		if (!labels) return;
		classesOpen = true;
		dockOpen = true;
		if (!classFormOpen) {
			className = "";
			pickedColor = "";
			classError = "";
			classAdded = "";
		}
		addingClass = true;
		// Someone may have added classes since this page loaded; the color offered is a fresh one.
		await labels.refreshClasses().catch(() => {});
		await tick();
		classInput?.focus();
	}

	/** Escape, from anywhere in the form, closes it (while there are classes to go back to) and does nothing else. */
	function closesOnEscape(form: HTMLFormElement) {
		const onKey = (event: KeyboardEvent) => {
			if (event.key !== "Escape" || event.isComposing || classes.length === 0) return;
			event.stopPropagation();
			void closeClassForm();
		};
		form.addEventListener("keydown", onKey);
		return () => form.removeEventListener("keydown", onKey);
	}

	/** Close the form, leaving keyboard focus on the button that opens it. */
	async function closeClassForm() {
		if (savingClass) return;
		addingClass = false;
		className = "";
		pickedColor = "";
		classError = "";
		await tick();
		addClassButton?.focus();
	}

	async function saveClass(event: SubmitEvent) {
		event.preventDefault();
		const name = className.trim();
		if (!labels || !name || savingClass) return;
		if (pickable.some((c) => c.name.trim().toLowerCase() === name.toLowerCase())) {
			classError = `There's a class called ${name} already.`;
			return;
		}
		savingClass = true;
		classError = "";
		const first = classes.length === 0;
		try {
			const made = await labels.addClass(name, classColor);
			// Ready to paint with it, which is why anyone adds one.
			viewer.activeClass = made.value;
			if (first) setTool("brush");
			classAdded = `Added ${made.name}.`;
			savingClass = false;
			await closeClassForm();
		} catch (e) {
			classError = message(e);
		} finally {
			savingClass = false;
		}
	}
	const zoomPercent = $derived(Math.round((viewer.zoom / (globalThis.devicePixelRatio || 1)) * 100));
	const [, imageZ, imageY, imageX] = manifest.shape_czyx;
	const histogram = manifest.histogram && !Array.isArray(manifest.histogram) ? manifest.histogram : null;
	const minimapSuggestions = $derived.by(() => live?.enabled
		? live.store ? [{ store: live.store, box: [0, 0, 0, ...viewer.shape] as Box, keys: [...live.regions.keys()] }] : []
		: [proposal, prediction].filter((source) => source !== null));
	const chip = $derived(saveState(queue));

	const nameOf = (values: number[]) => pickable.filter((c) => values.includes(c.value)).map((c) => c.name);
	/** Where painting or erasing goes under the mode, as a phrase to put after what the tool does ("" for anywhere). */
	const paintWhere = $derived(describeWhere(viewer.paintMode, nameOf(paintSet)));
	const eraseWhere = $derived(describeWhere(viewer.eraseMode, nameOf(eraseSet)));

	const hint = $derived(
		{
			navigate: "Drag to pan · wheel steps slices · Ctrl+wheel zooms · right-click moves the crosshair",
			brush: `Drag to paint the active class${paintWhere ? ` ${paintWhere}` : ""} · [ ] change the size · right-click moves the crosshair`,
			eraser: `Drag to erase labels${eraseWhere ? ` ${eraseWhere}` : ""} · [ ] change the size · right-click moves the crosshair`,
			accept: viewer.acceptShape === "brush"
				? "Drag to accept visible predictions · [ ] change size · Esc cancels · Ctrl+Z undoes the gesture"
				: "Click points or drag freehand · first point, double-click, or Enter accepts · Esc cancels · Ctrl+Z undoes",
			polygon:
				viewer.polygonMode === "add"
					? `Click points or drag freehand · click the first point, double-click, or Enter fills${paintWhere ? ` ${paintWhere}` : ""} · hold Alt to cut out · Esc cancels · right-click moves the crosshair`
					: "Click points or drag freehand · click the first point, double-click, or Enter cuts out the active class · hold Shift to fill · Esc cancels · right-click moves the crosshair",
			roi: "Drag a box on a slice · G goes to the next open ROI · C marks it complete · right-click moves the crosshair",
		}[viewer.tool],
	);
</script>

<svelte:window onkeydown={key} onkeyup={keyUp} onblur={blurred} />

{#snippet problem(text: string, dismiss: () => void)}
	<div
		class="pointer-events-auto flex w-full max-w-lg items-start gap-2 rounded-sm border border-danger/50 bg-panel px-2.5 py-1.5 text-danger shadow-lg shadow-black/40"
		role="alert"
	>
		<span class="flex-1">{text}</span>
		<button class="text-ink-dim hover:text-ink" aria-label="Dismiss" onclick={dismiss}><X size={14} /></button>
	</div>
{/snippet}

<div class="flex h-full flex-col bg-chrome text-ink">
	<!-- Options bar: the active tool's settings, wrapping onto more lines when the window is narrow. -->
	<div class="flex min-h-9 shrink-0 items-start bg-panel shadow-[inset_0_-1px_0_var(--color-edge)]" {@attach keepFocus}>
		<div class="flex min-w-0 flex-1 flex-wrap items-center gap-x-4 gap-y-1.5 px-3 py-1.5">
			<span class="flex h-6 shrink-0 items-center gap-1.5 font-medium">
				{#each TOOLS as entry (entry.tool)}
					{#if entry.tool === viewer.tool}<entry.icon size={14} class="text-ink-dim" />{entry.label}{/if}
				{/each}
			</span>
			<span class="h-4 w-px shrink-0 bg-line" aria-hidden="true"></span>
			{#if viewer.tool === "accept"}
				<Segmented label="Accept shape" value={viewer.acceptShape} options={ACCEPT_SHAPES} onchange={setAcceptShape} />
			{/if}
			{#if viewer.drawingBrush}
				<SliderField label="Size" numberLabel="Brush radius" min={0.5} max={64} step={0.5} bind:value={viewer.brushRadius} />
			{/if}
			{#if viewer.tool === "accept"}
				<Segmented label="Which predictions to accept" value={viewer.acceptMode} options={ACCEPT_MODES} onchange={setAcceptMode} />
				{#if viewer.acceptMode === "classes"}
					<ClassMenu classes={classes} selected={acceptSet} label="Accept predictions of" onchange={setAcceptClasses} />
				{/if}
				<span class="text-2xs text-ink-dim" role="status">
					{#if accepting}Accepting…
					{:else if acceptedVoxels !== null}{acceptedVoxels ? `Accepted ${acceptedVoxels.toLocaleString()} voxels · Ctrl+Z to undo` : "No matching, unannotated predictions in that selection"}
					{:else}Only unlabeled predictions · skips Background{/if}
					{#if acceptGesture && !accepting} · preview frozen{/if}
				</span>
			{/if}
			{#if viewer.tool === "polygon"}
				<Segmented label="Polygon mode" value={viewer.polygonMode} options={POLYGON_MODES} onchange={(mode) => (viewer.polygonMode = mode)} />
			{/if}
			{#if viewer.tool === "brush" || viewer.tool === "polygon"}
				{@const cuts = viewer.tool === "polygon" && viewer.polygonMode === "subtract"}
				<Segmented
					label="Where to paint"
					value={viewer.paintMode}
					options={PAINT_MODES}
					onchange={setPaintMode}
					class={cuts ? "opacity-60" : ""}
					title={cuts ? "These say where Add (and closing with Shift) fills. Subtract cuts only the active class out, whatever they say." : undefined}
				/>
				{#if viewer.paintMode === "classes"}
					<ClassMenu classes={pickable} selected={paintSet} label="Paint only over" onchange={(values) => (viewer.paintClasses = values)} />
				{/if}
			{/if}
			{#if viewer.tool === "eraser"}
				<Segmented label="What to erase" value={viewer.eraseMode} options={ERASE_MODES} onchange={setEraseMode} />
				{#if viewer.eraseMode === "classes"}
					<ClassMenu classes={pickable} selected={eraseSet} label="Erase only" onchange={(values) => (viewer.eraseClasses = values)} />
				{/if}
			{/if}
			{#if viewer.tool === "roi"}
				<label class="flex shrink-0 items-center gap-2 text-ink-dim">
					Depth
					<input class="w-28" type="range" min="1" max="256" step="1" bind:value={viewer.roiDepth} />
					<span class="w-24 font-mono text-ink">{viewer.roiDepth === 1 ? "slice" : `${viewer.roiDepth} voxels`}</span>
				</label>
			{/if}
			{#if viewer.tool === "navigate"}
				<Segmented label="Layout" value={viewer.layout} options={LAYOUT_OPTIONS} onchange={(layout) => (viewer.layout = layout)} />
				<button class="btn shrink-0" onclick={fit} use:tooltip={"Fit the image (0)"}><Maximize2 size={12} /> Fit</button>
			{/if}
			<span class="hidden min-w-48 flex-1 basis-48 truncate text-right text-2xs text-ink-faint xl:block" use:tooltip={hint}>{hint}</span>
		</div>
		<button
			class="btn btn-ghost mx-1.5 mt-1.5 shrink-0 md:hidden"
			aria-label="Panels"
			aria-expanded={dockOpen}
			onclick={() => (dockOpen = !dockOpen)}
		>
			<PanelRight size={14} />
		</button>
	</div>

	<div class="relative flex min-h-0 flex-1">
		<!-- Tools -->
		<div
			class="flex w-11 shrink-0 flex-col items-center gap-0.5 border-r border-edge bg-panel py-1.5"
			role="toolbar"
			aria-label="Tools"
			aria-orientation="vertical"
			{@attach keepFocus}
		>
			{#each TOOLS as entry (entry.tool)}
				<ToolButton
					icon={entry.icon}
					label={entry.label}
					shortcut={entry.shortcut}
					active={viewer.tool === entry.tool}
					onclick={() => setTool(entry.tool)}
				/>
			{/each}
			<span class="my-1.5 h-px w-6 bg-line"></span>
			<!-- The class brushes and polygons paint, like a foreground color. -->
			<button
				class="size-7 rounded-sm border-2 border-ink/80 shadow-[0_0_0_1px_black]"
				style:background={activeClass?.color ?? "transparent"}
				use:tooltip={activeClass ? `Painting ${activeClass.name} (1–9 to change)` : "Add a class to start labeling"}
				aria-label={activeClass ? `Active class: ${activeClass.name}` : "Add a class"}
				onclick={() => (activeClass ? ((classesOpen = true), (dockOpen = true)) : startClass())}
			></button>
			<span class="my-1.5 h-px w-6 bg-line"></span>
			<ToolButton icon={Undo2} label="Undo" shortcut="Ctrl+Z" disabled={queue.undoable === 0} onclick={() => historyStep()} />
			<ToolButton icon={Redo2} label="Redo" shortcut="Ctrl+Shift+Z" disabled={queue.redoable === 0} onclick={() => historyStep(true)} />
			<div class="mt-auto">
				<ToolButton icon={Keyboard} label="Keys" shortcut="?" active={viewer.help} onclick={() => (viewer.help = !viewer.help)} />
			</div>
		</div>

		<!-- Image views -->
		<div class="relative flex min-w-0 flex-1 flex-col">
			<!-- What went wrong, over the views, where it shows even with the dock closed. -->
			{#if error || predictionError || notice || rois.error}
				<div class="pointer-events-none absolute inset-x-0 top-8 z-30 flex flex-col items-center gap-1 px-2">
					{#if error}{@render problem(error, () => (error = ""))}{/if}
					{#if predictionError}{@render problem(predictionError, () => (predictionError = ""))}{/if}
					{#if notice}{@render problem(notice, () => (notice = ""))}{/if}
					{#if rois.error}{@render problem(rois.error, () => (rois.error = ""))}{/if}
				</div>
			{/if}
			<div
				class="grid min-h-0 flex-1 gap-px bg-edge
					{viewer.layout === 'four' ? 'grid-cols-2 grid-rows-2' : 'grid-cols-1 grid-rows-1'}"
			>
				{#if images && levels.length > 0}
					{#each shown as plane (plane.name)}
						<div
							class="contents"
							role="presentation"
							onpointerover={() => (pointed = plane)}
							onpointerout={(event) => {
								if (!event.currentTarget.contains(event.relatedTarget as Node | null)) pointed = null;
							}}
							onpointerdown={() => (used = plane)}
							onwheel={() => (used = plane)}
							onfocusin={() => (used = plane)}
						>
							<PlaneView
								{plane}
								{viewer}
								{levels}
								{images}
								{labels}
								classStyles={display.styles}
								paintColor={activeClass?.color}
								prediction={live?.enabled ? live.store : prediction?.store}
								predictionVersions={live?.enabled ? live.versions : undefined}
								stalePredictions={live?.enabled ? live.stale : undefined}
								animatePredictions={live?.updating ?? false}
								proposal={live?.enabled ? null : proposal}
								onhover={(p) => (hovered = p)}
								onresize={resized}
								onstroke={stroke}
								onpolygon={closePolygon}
								onacceptstart={beginAcceptGesture}
								onacceptcancel={cancelAcceptGesture}
								acceptGestureId={acceptGesture?.id ?? null}
								rois={rois.items}
								onroi={drawRoi}
							/>
						</div>
					{/each}
					{#if viewer.layout === "four"}
						<MiniMap {project} {viewer} {labels} classStyles={display.styles} suggestions={minimapSuggestions} onpanstart={() => cancelAcceptGesture()} />
					{/if}
				{:else}
					<div class="col-span-full row-span-full grid place-items-center bg-pasteboard text-ink-faint">
						{#if error}<ImageOff size={20} />{:else}<LoaderCircle size={20} class="animate-spin" />{/if}
					</div>
				{/if}
			</div>
		</div>

		<!-- Dock -->
		<aside
			class="{dockOpen ? 'flex' : 'hidden'} absolute inset-y-0 right-0 z-20 w-64 shrink-0 flex-col overflow-y-auto border-l border-edge bg-panel shadow-2xl shadow-black/50 md:static md:flex md:shadow-none"
			aria-label="Panels"
			{@attach keepFocus}
		>
			<Panel title="Info">
				<div class="grid grid-cols-[1rem_1fr] gap-x-1 gap-y-0.5 font-mono text-2xs">
					<span class="text-axis-x">X</span><span>{voxel(2)}</span>
					<span class="text-axis-y">Y</span><span>{voxel(1)}</span>
					<span class="text-axis-z">Z</span><span>{voxel(0)}</span>
				</div>
			</Panel>

			<Panel title="Levels">
				{#if histogram}
					<Histogram counts={histogram.counts} edges={histogram.edges} bind:window={viewer.window} />
				{/if}
				<!-- Side by side, or one over the other when what they show needs the room. -->
				<div class="flex flex-wrap gap-2">
					<label class="label min-w-0 flex-1">Black <input class="field w-full font-mono" type="number" step="1" bind:value={() => displayLevel(viewer.window[0]), (value) => viewer.window = editLevel(viewer.window, 0, value)} /></label>
					<label class="label min-w-0 flex-1">White <input class="field w-full font-mono" type="number" step="1" bind:value={() => displayLevel(viewer.window[1]), (value) => viewer.window = editLevel(viewer.window, 1, value)} /></label>
				</div>
			</Panel>

			<Panel title="Label suggestions">
				<LivePanel {live} plugins={modelPlugins} disabled={decidingSuggestion || !!pendingAccept}
					modelsHref={`/p/${project}/models`} ontoggle={toggleLive} onshow={() => viewer.showPrediction = true} />
			</Panel>

			<Panel title="Layers">
				<ul class="-mx-2.5 -my-2.5 flex flex-col divide-y divide-edge">
					<li class="flex flex-col gap-1.5 px-2.5 py-2">
						<div class="flex items-center gap-2">
							<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showLabels ? 'Hide' : 'Show'} labels" use:tooltip={"Show or hide (V)"} onclick={() => (viewer.showLabels = !viewer.showLabels)}>
								{#if viewer.showLabels}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
							</button>
							<span class="flex-1">Saved labels</span>
							<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.opacity * 100)}%</span>
						</div>
						<span class="pl-5.5 text-2xs text-ink-dim">Solid · painted or kept with Accept</span>
						<input type="range" min="0" max="1" step="0.05" bind:value={viewer.opacity} aria-label="Segmentation opacity" />
					</li>
					{#if prediction || proposal || live?.enabled}
						<li class="flex flex-col gap-1.5 px-2.5 py-2">
							<div class="flex items-center gap-2">
								<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showPrediction ? 'Hide' : 'Show'} suggestions" use:tooltip={live?.enabled ? `${viewer.showPrediction ? 'Hide suggestions and pause automatic updates' : 'Show suggestions and resume automatic updates'} (M)` : "Show or hide suggestions (M)"} onclick={() => (viewer.showPrediction = !viewer.showPrediction)}>
									{#if viewer.showPrediction}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
								</button>
								<span class="flex-1 truncate">
									Suggestions
								</span>
							</div>
					<p class="pl-5.5 text-2xs text-ink-dim">Predicted for review (striped)</p>
							{#if prediction && proposal && !live?.enabled}
								<span class="truncate pl-5.5 text-2xs text-ink-dim">
									Proposal in one ROI <span class="text-ink-faint">· {proposal.model_name ?? "a model"}</span>
								</span>
							{/if}
							{#if labels}
								{@const { plane, box, layer, blocked } = inView}
								{@const kind = (layer ?? prediction ?? proposal)?.kind ?? "prediction"}
								<!-- A goes to the selected ROI, if one is, so it's this button's key only without one. -->
								{@const keyHint = selectedRoi ? "" : " (A)"}
								<div class="flex flex-col gap-1.5">
									<button
										class="btn h-auto min-h-6 min-w-0 flex-1 py-1.5"
										disabled={decidingSuggestion || !!blocked}
										use:tooltip={blocked ||
											`Review the unlabeled voxels in the visible part of this slice (${plane.name.toUpperCase()} view, ${"zyx"[plane.normal]} ${box?.[plane.normal]}) before accepting the ${kind}; you can undo it${keyHint}`}
										aria-describedby={blocked && !decidingSuggestion ? "accept-view-why" : undefined}
										onclick={acceptView}
									>
										<CheckCheck size={13} class="shrink-0" />
										<span class="min-w-0 flex-1 text-left leading-4">{accepting ? "Preparing review…" : `Accept ${plane.name.toUpperCase()} suggestions for visible slice`}</span>
										{#if !selectedRoi}<span class="kbd ml-auto">A</span>{/if}
									</button>
									<button
										class="btn min-w-0 flex-1"
										disabled={decidingSuggestion || !!blocked}
										use:tooltip={blocked || `Hide the ${kind}'s foreground suggestions in this view without training them as Background; you can undo it`}
										onclick={declineView}
									>
										<X size={13} />
										{declining ? "Declining…" : `Decline ${plane.name.toUpperCase()} suggestions`}
										{#if !selectedRoi}<span class="kbd ml-auto">X</span>{/if}
									</button>
								</div>
								<button
									class="btn w-full"
									disabled={decidingSuggestion || !!blocked}
									use:tooltip={blocked || "Reveal suggestions explicitly declined in this view; newer labels are left alone"}
									onclick={restoreView}
								>
									<RotateCcw size={13} />
									{restoring ? "Restoring…" : `Restore declined in ${plane.name.toUpperCase()}`}
								</button>
								{#if blocked && !decidingSuggestion}
									<p id="accept-view-why" class="text-2xs text-ink-dim">{blocked}</p>
								{/if}
							{/if}
						</li>
					{/if}
					<li class="flex items-center gap-2 px-2.5 py-2 text-ink-dim">
						<Eye size={14} class="opacity-40" />
						<span class="flex-1">Image</span>
						<Lock size={12} />
					</li>
				</ul>
			</Panel>

			<Panel title="Classes" bind:open={classesOpen}>
				{#snippet actions()}
					{#if labels && !classFormOpen}
						<button class="btn btn-ghost gap-1" bind:this={addClassButton} title="Add a class to label with" onclick={startClass}>
							<Plus size={12} /> Add class
						</button>
					{/if}
				{/snippet}
				{#if labels && classes.length === 0}
					<div class="flex flex-col gap-1 rounded-sm border border-edge bg-field p-2.5">
						<p class="font-medium text-ink">Start labeling</p>
						<p class="text-ink-dim">
							The model learns from two kinds of labels: what you're looking for, and background, which is everything else.
						</p>
						<ol class="ml-4 list-decimal text-ink-dim">
							<li>Name what you're looking for, below.</li>
							<li>Paint a little of it, then paint some Background, which is already in the list.</li>
						</ol>
					</div>
				{/if}
				<ul class="-mx-2.5 -my-1 flex flex-col">
					{#each pickable as label, index (label.value)}
						<ClassRow {label} {index} {display} active={viewer.activeClass === label.value}
							disabled={!labels || decidingSuggestion || erasingComposite || !!pendingAccept}
							onselect={() => viewer.activeClass = label.value}
							oncolor={(color) => previewClassColor(label.value, color)}
							onsave={async () => { if (label.value !== BACKGROUND_VALUE) await labels?.saveColor(label.value); }} />
					{/each}
				</ul>
				{#if labels && classes.length > 0 && !hasBackground}
					<div class="flex flex-col gap-1.5 text-ink-dim">
						<p>
							<span class="text-ink">Paint some Background too:</span> everything that isn't what you're looking for. The model needs both to tell them apart.
						</p>
						<button
							class="btn self-start"
							onclick={() => {
								viewer.activeClass = BACKGROUND_VALUE;
								if (viewer.tool !== "brush" && viewer.tool !== "polygon") setTool("brush");
							}}
						>
							Paint Background
						</button>
					</div>
				{/if}
				<p class="sr-only" role="status">{classAdded}</p>
				{#if classFormOpen}
					<form class="flex flex-col gap-2" onsubmit={saveClass} {@attach closesOnEscape}>
						<div class="flex items-center gap-2">
							<input
								class="field"
								bind:this={classInput}
								bind:value={className}
								maxlength="100"
								placeholder={classes.length === 0 ? "What are you looking for? Like bone" : "Class name, like bone"}
								aria-label="Class name"
								autocomplete="off"
								required
							/>
							<input
								type="color"
								class="h-6 w-8 shrink-0 cursor-pointer rounded-sm border border-edge bg-field p-0.5"
								value={classColor}
								oninput={(event) => (pickedColor = event.currentTarget.value)}
								aria-label="Class color"
								title="Class color"
							/>
						</div>
						{#if classError}<p class="error" role="alert">{classError}</p>{/if}
						<div class="flex items-center gap-2">
							<button class="btn btn-primary" disabled={!className.trim()} aria-disabled={savingClass}>{classes.length === 0 ? "Start labeling" : "Add"}</button>
							{#if classes.length > 0}
								<button type="button" class="btn btn-ghost" disabled={savingClass} onclick={closeClassForm}>Cancel</button>
							{/if}
						</div>
					</form>
				{/if}
			</Panel>

			{#if SHOW_ROIS}
			<Panel title="ROIs · {openRois.length} open">
				<ul class="-mx-2.5 -my-1 flex max-h-52 flex-col overflow-y-auto">
					{#each rois.items as roi (roi.id)}
						<li class="flex items-center gap-1.5 px-2.5 py-0.5 {roi.id === viewer.selectedRoi ? 'bg-accent-soft' : 'hover:bg-raised'}">
							<button class="flex flex-1 items-center gap-1.5 truncate text-left" onclick={() => goTo(roi)}>
								<span class="size-1.5 shrink-0 rounded-full {roi.status === 'complete' ? 'bg-ok' : roi.status === 'skipped' ? 'bg-ink-faint' : 'bg-warn'}"></span>
								<span class="truncate">{describe(roi)}</span>
							</button>
							<select
								class="field !h-5 !w-20 !text-2xs"
								aria-label="Status of the {describe(roi)} ROI"
								value={roi.status}
								onchange={async (e) => {
									const select = e.currentTarget;
									if (!(await rois.update(roi.id, { status: select.value as Roi["status"] }))) revert(select, roi.status);
								}}
							>
								<option value="open">open</option>
								<option value="complete">complete</option>
								<option value="skipped">skipped</option>
							</select>
						</li>
					{:else}
						<li class="px-2.5 text-ink-dim">Draw one with the ROI tool (R).</li>
					{/each}
				</ul>
				{#if (proposer && selectedRoi) || proposing}
					<div class="flex gap-1.5">
						<button
							class="btn flex-1"
							disabled={proposing || !!proposeBlocked}
							title={proposeBlocked || `Predict just this ROI with ${proposer?.name ?? "the newest model"}, ahead of other work`}
							onclick={() => selectedRoi && propose(selectedRoi)}
						>
							<Sparkles size={13} />
							{proposeLabel}
						</button>
						{#if proposing && proposalPipeline}
							<button
								class="btn"
								disabled={cancelling}
								aria-label="Cancel the proposal"
								title={cancelling ? "Stopping…" : "Stop making this proposal"}
								onclick={cancelProposal}
							>
								<X size={13} />
							</button>
						{/if}
					</div>
					{/if}
				{#if shownHere && selectedRoi}
					<div class="flex gap-1.5">
						<button
							class="btn flex-1"
							disabled={decidingSuggestion || !!acceptBlocked}
							use:tooltip={acceptBlocked || `Review the selected ROI's unlabeled voxels before accepting the ${shownHere.kind} (A)`}
							onclick={() => selectedRoi && acceptPrediction(selectedRoi)}
						>
							<CheckCheck size={13} />
							{accepting ? "Reading…" : "Accept"}<span class="kbd ml-auto">A</span>
						</button>
						<button
							class="btn flex-1"
							disabled={decidingSuggestion || !!acceptBlocked}
							use:tooltip={acceptBlocked || `Decline the ${shownHere.kind}'s foreground suggestions in the selected ROI (X)`}
							onclick={() => selectedRoi && declinePrediction(selectedRoi)}
						>
							<X size={13} />
							{declining ? "Declining…" : "Decline"}<span class="kbd ml-auto">X</span>
						</button>
						</div>
						<button
							class="btn w-full"
							disabled={decidingSuggestion || !!acceptBlocked}
							title={acceptBlocked || "Reveal suggestions explicitly declined in the selected ROI; newer labels are left alone"}
							onclick={() => selectedRoi && restorePrediction(selectedRoi)}
						>
							<RotateCcw size={13} />
							{restoring ? "Restoring…" : "Restore declined"}
						</button>
					{/if}
				<div class="flex items-center gap-2">
					<button
						class="btn"
						disabled={exploring}
						title="Make an ROI at a random place no ROI covers and go there; with a ready model, propose labels there"
						onclick={explore}
					>
						<Compass size={13} />
						{exploring ? "Exploring…" : "Explore"}
					</button>
					<a href="/p/{project}/rois" class="ml-auto text-2xs">Open the ROI gallery</a>
				</div>
			</Panel>
			{/if}
		</aside>
	</div>

	<!-- Status bar -->
	<footer class="flex min-h-6 shrink-0 flex-wrap items-center gap-x-4 border-t border-edge bg-chrome px-3 py-1 font-mono text-2xs text-ink-dim" aria-label="Image status">
		<span title="Zoom">{zoomPercent}%</span>
		<span class="whitespace-nowrap">
			<span class="text-axis-x">x</span>{Math.floor(viewer.position[2])}
			<span class="text-axis-y">y</span>{Math.floor(viewer.position[1])}
			<span class="text-axis-z">z</span>{Math.floor(viewer.position[0])}
		</span>
		<span class="hidden sm:inline">{LAYOUT_NAMES[viewer.layout]}</span>
		<span class="whitespace-nowrap" title="Image dimensions (X × Y × Z) in voxels, and data type">{imageX}×{imageY}×{imageZ} · {manifest.dtype}</span>
		<span class="ml-auto flex items-center gap-1.5 font-sans {chip === 'error' ? 'text-danger' : chip === 'offline' || chip === 'retrying' ? 'text-warn' : ''}" role="status">
			{#if chip === "saved"}<Check size={12} class="text-ok" />{:else if chip === "saving"}<LoaderCircle size={12} class="animate-spin" />{:else}<CloudOff size={12} />{/if}
			{status}
		</span>
	</footer>
</div>

{#if pendingAccept}
	<!-- svelte-ignore a11y_click_events_have_key_events: Escape is handled by the window key handler. -->
	<div class="fixed inset-0 z-40 grid place-items-center bg-black/50 p-4" role="presentation" onclick={cancelAccept}>
		<div
			class="panel w-full max-w-md shadow-2xl shadow-black/60"
			role="dialog"
			aria-modal="true"
			aria-labelledby="accept-title"
			tabindex="-1"
			{@attach holdFocus}
			onclick={(event) => event.stopPropagation()}
		>
			<div class="panel-title" id="accept-title">Accept suggestions in {pendingAccept.place}?</div>
			<div class="flex flex-col gap-3 p-3">
				<p class="text-ink-dim">
					This saves ready suggestions only where nothing is labeled yet. Your saved labels stay unchanged. Background is excluded unless you check it below.
				</p>
				<div class="rounded-sm border border-edge bg-field px-2.5 py-2">
					<div class="flex items-center justify-between gap-3">
						<span>Suggested labels (not Background)</span>
						<span class="font-mono text-2xs">{pendingAccept.classes.toLocaleString()} voxels</span>
					</div>
					<label class="mt-2 flex cursor-pointer items-center justify-between gap-3 border-t border-edge pt-2">
						<span class="flex items-center gap-2">
							<input type="checkbox" bind:checked={includeAcceptBackground} disabled={pendingAccept.background === 0} />
							Also Background
						</span>
						<span class="font-mono text-2xs">{pendingAccept.background.toLocaleString()} voxels</span>
					</label>
				</div>
				{#if pendingAccept.classes === 0}
					<p class="text-warn">Only Background suggestions remain here. Check Also Background to save them, or cancel.</p>
				{/if}
				<div class="flex justify-end gap-2">
					<button class="btn" onclick={cancelAccept}>Cancel</button>
					<button class="btn btn-primary" disabled={pendingAccept.classes === 0 && !includeAcceptBackground} onclick={confirmAccept}>
						<CheckCheck size={13} />
						Accept {(
							pendingAccept.classes + (includeAcceptBackground ? pendingAccept.background : 0)
						).toLocaleString()} voxels
						<span class="kbd">A</span>
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

{#if viewer.help}
	<div class="fixed inset-0 z-40 grid place-items-center bg-black/50 p-4" role="presentation" onclick={() => (viewer.help = false)}>
		<div
			class="panel max-h-[80vh] w-full max-w-lg overflow-auto shadow-2xl shadow-black/60"
			role="dialog"
			aria-modal="true"
			aria-label="Keys"
			tabindex="-1"
			{@attach holdFocus}
			onclick={(e) => e.stopPropagation()}
			onkeydown={helpKey}
		>
			<div class="panel-title">Keyboard shortcuts</div>
			<table class="w-full">
				<tbody>
					{#each KEYMAP as binding (binding.action)}
						<tr class="border-b border-edge">
							<td class="px-3 py-1.5 whitespace-nowrap">
								{#each binding.keys as k (k)}<span class="kbd mr-1">{k.replace("mod+", "Ctrl+")}</span>{/each}
							</td>
							<td class="px-3 py-1.5 text-ink-dim">{binding.label}</td>
						</tr>
					{/each}
					{#each MOUSE as [what, does] (what)}
						<tr class="border-b border-edge">
							<td class="px-3 py-1.5">{what}</td>
							<td class="px-3 py-1.5 text-ink-dim">{does}</td>
						</tr>
					{/each}
				</tbody>
			</table>
			<div class="flex justify-end p-2">
				<button class="btn" onclick={() => (viewer.help = false)}>Close</button>
			</div>
		</div>
	</div>
{/if}
