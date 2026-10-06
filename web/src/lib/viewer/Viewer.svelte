<script lang="ts">
	import Brush from "@lucide/svelte/icons/brush";
	import Check from "@lucide/svelte/icons/check";
	import CheckCheck from "@lucide/svelte/icons/check-check";
	import CloudOff from "@lucide/svelte/icons/cloud-off";
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
	import Redo2 from "@lucide/svelte/icons/redo-2";
	import Sparkles from "@lucide/svelte/icons/sparkles";
	import SquareDashed from "@lucide/svelte/icons/square-dashed";
	import Undo2 from "@lucide/svelte/icons/undo-2";
	import X from "@lucide/svelte/icons/x";
	import { onDestroy, onMount, untrack } from "svelte";
	import Histogram from "#lib/ui/Histogram.svelte";
	import Panel from "#lib/ui/Panel.svelte";
	import ToolButton from "#lib/ui/ToolButton.svelte";
	import { ApiError, api, message } from "#lib/api.ts";
	import { unfinished } from "#lib/pipelines.ts";
	import { whileVisible } from "#lib/refresh.ts";
	import { session } from "#lib/session.svelte.ts";
	import type { Pipeline, ProjectImage } from "#lib/types.ts";
	import { acceptParts, MAX_ACCEPT_VOXELS, readBox } from "../labels/accept";
	import { splitIntoDeltas } from "../labels/deltas";
	import { indexedDbStorage, OpQueue, type QueuedEdit } from "../labels/opqueue.svelte";
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
	import { ChunkStore } from "./chunks";
	import { absolute, loadLevels } from "./image";
	import { actionFor, forFocused, KEYMAP, MOUSE } from "./keymap";
	import { type LabelClass, LabelLayer } from "./labels";
	import { imageLoader, labelLoader, WorkerPool } from "./loader";
	import PlaneView from "./PlaneView.svelte";
	import { LAYOUTS, type Stroke, ViewerState } from "./state.svelte";
	import { aspectOf, type Level, type Plane, PLANES, type Vec3 } from "./tiles";

	let {
		image,
		projectId,
		title = "Image",
		roi: startRoi = null,
	}: { image: ProjectImage; projectId: string; title?: string; roi?: string | null } = $props();

	const CACHE_BYTES = 512 * 1024 * 1024;
	// The most one proposal predicts (the server's limit).
	const MAX_PROPOSAL_VOXELS = 256 ** 3;

	// The page makes a new viewer for each image.
	const { manifest, zarr_url: zarrUrl } = untrack(() => image);
	const project = untrack(() => projectId);
	const [, nz, ny, nx] = manifest.shape_czyx;
	const viewer = new ViewerState([nz, ny, nx], aspectOf(manifest.voxel_size_zyx));
	viewer.window = [...manifest.window];
	viewer.position = [nz / 2, ny / 2, nx / 2];

	let levels: Level[] = $state([]);
	let images: ChunkStore | null = $state(null);
	let labels: LabelLayer | null = $state(null);
	let prediction: Prediction | null = $state.raw(null);
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
	let segmentation: ChunkStore | null = $state(null);
	let classes: LabelClass[] = $state([]);
	let error = $state("");
	// Why the prediction layer couldn't load when the page opened; cleared
	// when a later load works.
	let predictionError = $state("");
	let pool: WorkerPool | undefined;
	let hovered: Plane = PLANES.xy;
	let notice = $state("");
	let classesOpen = $state(true);
	// On narrow screens the dock floats over the views until closed.
	let dockOpen = $state(false);
	const me = session.current?.user.id ?? "";
	const queue = new OpQueue(project, indexedDbStorage(me, project));
	// Strict edits compare against the chunk versions current when they're
	// sent, after this page's earlier edits have landed; if any version is
	// unknown, the edit applies like a brush stroke instead.
	queue.beforeSend = (op: QueuedEdit) => {
		if (!op.strict || !labels) return op;
		const versions = op.deltas.map((d) => labels!.versionOf(d.key.join("/")));
		if (versions.some((v) => v === undefined)) return { ...op, strict: false };
		return { ...op, deltas: op.deltas.map((d, i) => ({ ...d, base_version: versions[i]! })) };
	};
	const rois = new RoiList(project);
	const firstRoi = untrack(() => startRoi);
	const sizes = new Map<string, [number, number]>();
	const controller = new AbortController();

	const voxelSize = manifest.voxel_size_zyx;
	const unit = manifest.unit ?? "";

	onMount(async () => {
		try {
			levels = await loadLevels(zarrUrl, controller.signal);
			rois.load().then(() => {
				const found = rois.items.find((r) => r.id === firstRoi);
				if (found) goTo(found);
				else if (firstRoi) rois.error = "That ROI isn't in this project any more.";
			});
			pool = new WorkerPool();
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
			api<{ zarr_url: string }>(`/api/projects/${project}/segmentation`).then(
				(found) => {
					if (!pool || controller.signal.aborted) return;
					segmentation = new ChunkStore(labelLoader(pool, absolute(found.zarr_url), viewer.shape), 128 * 1024 * 1024, 4);
				},
				(e: unknown) => {
					if (!(e instanceof ApiError && e.status === 404)) error = e instanceof Error ? e.message : String(e);
				},
			);
			const layer = new LabelLayer(project, pool, viewer.shape);
			await layer.start();
			if (controller.signal.aborted) return layer.stop();
			layer.onStopped = () => (error = "Live label updates stopped. Reload the page to see others' edits.");
			labels = layer;
			classes = layer.classes;
			viewer.activeClass ??= classes[0]?.value ?? null;
			queue.onOutcome((outcome) => {
				if ("cancelled" in outcome) {
					layer.settle(outcome.op.local, null);
				} else if (outcome.op.kind !== "edit") {
					if ("result" in outcome) layer.noteVersions(outcome.result.chunks);
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

	const stopRefreshing = rois.keepFresh();
	// Models trained or deleted, and predictions and proposals made, since.
	// A reload that fails keeps what's shown, quietly, and the next one tries again.
	const stopReloading = whileVisible(() => void loadPrediction(controller.signal).catch(() => {}));

	onDestroy(() => {
		controller.abort();
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
			viewer.showLabels,
			viewer.layout,
			viewer.brushRadius,
			viewer.protectLabels,
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
		if (pendingGoTo) return goTo(pendingGoTo);
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
	}

	function stroke(drawn: Stroke) {
		if (!drawn.erase && drawn.value === 0) return;
		commit(drawn.plane, drawn.slice, drawn.mask, drawn.value, drawn.onlyIf, {
			name: drawn.erase ? "eraser" : "brush",
			radius: drawn.radius,
			plane: drawn.plane.name,
			slice: drawn.slice,
		});
	}

	/** Fill the polygon being drawn; `erase` clears the active class inside it instead. */
	function closePolygon(erase = false) {
		const polygon = viewer.polygon;
		viewer.polygon = null;
		if (!polygon || polygon.points.length < 3 || viewer.activeClass === null) return;
		const plane = PLANES[polygon.plane];
		const mask = new PlaneMask(viewer.shape[plane.u], viewer.shape[plane.v]);
		mask.polygon(polygon.points);
		if (mask.count === 0) return;
		const value = erase ? 0 : viewer.activeClass;
		const onlyIf = erase ? `class:${viewer.activeClass}` : viewer.protectLabels ? "unlabeled" : "any";
		const points = polygon.points.map(([u, v]) => [Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
		const tool = { name: erase ? "polygon-erase" : "polygon", plane: plane.name, slice: polygon.slice, points: points.length <= 500 ? points : undefined };
		commit(plane, polygon.slice, mask, value, onlyIf, tool, true);
	}

	// --- ROIs ----------------------------------------------------------------

	async function drawRoi(plane: Plane, slice: number, corners: [[number, number], [number, number]]) {
		const bbox = roiBox(plane, slice, corners, viewer.roiDepth, viewer.shape);
		if (!bbox) return;
		const roi = await rois.add(bbox, viewer.roiDepth === 1 ? "slice" : "cube");
		if (roi) viewer.selectedRoi = roi.id;
	}

	// An ROI to fit once the main view knows its size.
	let pendingGoTo: Roi | null = null;

	/**
	 * Center the views on an ROI and fit it: a slice ROI in the view of its
	 * plane, a cube in every view shown.
	 */
	function goTo(roi: Roi) {
		viewer.selectedRoi = roi.id;
		viewer.autoFit = false;
		const { bbox } = roi;
		const thin = thinAxis(bbox);
		const slicePlane = thin === 0 ? PLANES.xy : thin === 1 ? PLANES.xz : PLANES.yz;
		viewer.moveTo([0, 1, 2].map((a) => (bbox[a]! + bbox[a + 3]!) / 2) as Vec3);
		if (roi.kind === "slice" && viewer.layout !== "four" && viewer.layout !== slicePlane.name) {
			// The new view's size arrives when it lays out; fit then.
			viewer.layout = slicePlane.name;
			sizes.clear();
			pendingGoTo = roi;
			return;
		}
		const planes = roi.kind === "slice" ? [slicePlane] : shown;
		const extent = (axis: number) => (bbox[axis + 3]! - bbox[axis]!) * viewer.aspect[axis]!;
		const zooms = planes.flatMap((plane) => {
			const size = sizes.get(plane.name);
			return size ? [Math.min(size[0] / extent(plane.u), size[1] / extent(plane.v))] : [];
		});
		pendingGoTo = zooms.length === planes.length ? null : roi;
		if (zooms.length > 0) viewer.zoom = Math.min(64, Math.max(1 / 512, 0.85 * Math.min(...zooms)));
	}

	/** A prediction, as the server describes it; a proposal's also has its box. */
	interface Predicted {
		artifact_id: string;
		zarr_url: string;
		model_id: string | null;
		model_name: string | null;
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
		if (signal.aborted || load !== loads) return;
		predictionError = "";
		// Newest first.
		if (models) proposer = models.find((m) => m.status === "ready") ?? null;
		// One of another image (one that replaced this since) wouldn't line up.
		const fits = (found: Predicted | null) => (found?.shape_zyx.join() === viewer.shape.join() ? found : null);
		const current = fits(whole);
		const newer = fits(proposed);
		const image: Box = [0, 0, 0, ...viewer.shape];
		prediction = show(prediction, current && { ...current, box: image }, "prediction");
		// A proposal asked for before the prediction is out of date; one asked
		// for after it shows, even if the prediction was done later.
		proposal = show(
			proposal,
			newer?.box && (!current || Date.parse(newer.started_at) > Date.parse(current.started_at)) ? newer : null,
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
	 * The layer an ROI shows, which accepting there reads: the proposal if the
	 * ROI is inside its box, else the prediction.
	 */
	function covering(roi: Roi): Prediction | null {
		const box = inImage(roi);
		return box ? ([proposal, prediction].find((layer) => layer && within(box, layer.box)) ?? null) : null;
	}

	/**
	 * Why accepting in an ROI can't go ahead, if it can't: where it reaches
	 * past the proposal's box, part of it shows the proposal and part the
	 * prediction, and accepting reads only one of them (unless the same
	 * model made both, so they agree).
	 */
	function acceptBlockedIn(roi: Roi): string {
		const box = inImage(roi);
		if (!box || !proposal || !overlaps(box, proposal.box) || within(box, proposal.box)) return "";
		if (prediction && prediction.model_id === proposal.model_id) return "";
		return "Part of this ROI shows your proposal and part doesn't; propose the whole ROI to accept it";
	}

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

	/**
	 * Copy the model's prediction inside an ROI into the labels, as accepted
	 * (model-verified) labels, without touching voxels anyone labeled. It
	 * reads what the ROI shows: the proposal, if the ROI is inside its box,
	 * else the prediction.
	 */
	async function acceptPrediction(roi: Roi) {
		if (!labels || accepting) return;
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
		accepting = true;
		notice = "";
		const { store, artifact_id: artifact, kind } = layer;
		try {
			const values = await readBox((id) => {
				// Keep these loads from being cancelled by the views' own requests.
				store.want(`accept:${id}`, new Set([id]));
				return store.request(id).finally(() => store.want(`accept:${id}`, new Set()));
			}, box);
			const parts = acceptParts(values, box);
			const ops = queue.editMany(parts, { accept: { prediction: artifact, roi: roi.id } });
			for (const op of ops) labels.applyLocal(op.local, op.deltas);
			if (ops.length === 0) notice = `The ${kind} has nothing in that ROI.`;
		} catch (e) {
			notice = `Couldn't read the ${kind}: ${e instanceof Error ? e.message : String(e)}`;
		} finally {
			accepting = false;
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
		viewer.tool = tool;
		if (tool !== "polygon") viewer.polygon = null;
	}

	const status = $derived(
		queue.error
			? queue.error
			: queue.offline
				? `Offline · ${queue.pending} waiting`
				: queue.pending > 0
					? `Saving ${queue.pending}`
					: "Saved",
	);

	function keyUp(event: KeyboardEvent) {
		if (event.key === " ") viewer.panning = false;
	}

	/** The view that slice keys step: the only one, or the one last pointed at. */
	function activePlane(): Plane {
		return viewer.layout === "four" ? hovered : PLANES[viewer.layout];
	}

	function key(event: KeyboardEvent) {
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
		// Enter and Backspace only mean something while drawing a polygon.
		if ((action === "close-polygon" || action === "remove-point") && !viewer.polygon) return;
		event.preventDefault();
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
			case "accept":
				if (selectedRoi) void acceptPrediction(selectedRoi);
				return;
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
				const chosen = classes[Number(event.key) - 1];
				if (chosen) viewer.activeClass = chosen.value;
				return;
			}
			case "close-polygon":
				return closePolygon(event.altKey);
			case "remove-point":
				if (viewer.polygon) viewer.polygon = { ...viewer.polygon, points: viewer.polygon.points.slice(0, -1) };
				return;
			case "cancel":
				if (viewer.polygon) viewer.polygon = null;
				else setTool("navigate");
				return;
			case "undo":
				queue.undo();
				return;
			case "redo":
				queue.redo();
				return;
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

	const TOOLS = [
		{ tool: "navigate", label: "Navigate", shortcut: "N", icon: Hand },
		{ tool: "brush", label: "Brush", shortcut: "B", icon: Brush },
		{ tool: "eraser", label: "Eraser", shortcut: "E", icon: Eraser },
		{ tool: "polygon", label: "Polygon", shortcut: "P", icon: Pentagon },
		{ tool: "roi", label: "ROI", shortcut: "R", icon: SquareDashed },
	] as const;

	const LAYOUT_NAMES = { four: "Four views", xy: "XY", xz: "XZ", yz: "YZ" } as const;

	const activeClass = $derived(classes.find((c) => c.value === viewer.activeClass));
	const zoomPercent = $derived(Math.round((viewer.zoom / (globalThis.devicePixelRatio || 1)) * 100));
	const [, imageZ, imageY, imageX] = manifest.shape_czyx;
	const histogram = manifest.histogram && !Array.isArray(manifest.histogram) ? manifest.histogram : null;
	const saveState = $derived(queue.error ? "error" : queue.offline ? "offline" : queue.pending > 0 ? "saving" : "saved");

	const hint = $derived(
		{
			navigate: "Drag to pan · wheel steps slices · Ctrl+wheel zooms · click moves the crosshair",
			brush: "Drag to paint the active class · [ ] change the size",
			eraser: "Drag to erase labels · [ ] change the size",
			polygon: "Click to add points · Enter or double-click fills · Alt+Enter erases the class inside · Esc cancels",
			roi: "Drag a box on a slice · G goes to the next open ROI · C marks it complete",
		}[viewer.tool],
	);
</script>

<svelte:window onkeydown={key} onkeyup={keyUp} onblur={() => (viewer.panning = false)} />

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
	<!-- Options bar: the active tool's settings, scrolling sideways when they don't fit. -->
	<div class="flex h-9 shrink-0 items-center border-b border-edge bg-panel" {@attach keepFocus}>
		<div class="flex min-w-0 flex-1 items-center gap-3 self-stretch overflow-x-auto px-3 whitespace-nowrap">
			<span class="flex shrink-0 items-center gap-1.5 font-medium">
				{#each TOOLS as entry (entry.tool)}
					{#if entry.tool === viewer.tool}<entry.icon size={14} class="text-ink-dim" />{entry.label}{/if}
				{/each}
			</span>
			<span class="h-4 w-px shrink-0 bg-line"></span>
			{#if viewer.tool === "brush" || viewer.tool === "eraser"}
				<label class="flex shrink-0 items-center gap-2 text-ink-dim">
					Size
					<input class="w-28" type="range" min="0.5" max="64" step="0.5" bind:value={viewer.brushRadius} />
					<input class="field w-14 font-mono" type="number" min="0.5" max="64" step="0.5" bind:value={viewer.brushRadius} aria-label="Brush radius" />
				</label>
			{/if}
			{#if viewer.tool === "brush" || viewer.tool === "polygon"}
				<label class="flex shrink-0 items-center gap-1.5 text-ink-dim">
					<input type="checkbox" bind:checked={viewer.protectLabels} /> Only unlabeled voxels
				</label>
			{/if}
			{#if viewer.tool === "roi"}
				<label class="flex shrink-0 items-center gap-2 text-ink-dim">
					Depth
					<input class="w-28" type="range" min="1" max="256" step="1" bind:value={viewer.roiDepth} />
					<span class="w-24 font-mono text-ink">{viewer.roiDepth === 1 ? "slice" : `${viewer.roiDepth} voxels`}</span>
				</label>
			{/if}
			{#if viewer.tool === "navigate"}
				<div class="flex shrink-0 rounded-sm border border-edge" role="group" aria-label="Layout">
					{#each LAYOUTS as layout (layout)}
						<button
							aria-pressed={viewer.layout === layout}
							class="relative flex h-6 items-center gap-1 px-2 first:rounded-l-[3px] last:rounded-r-[3px] focus-visible:z-10
								{viewer.layout === layout ? 'bg-accent-fill text-white' : 'bg-raised text-ink-dim hover:bg-hover hover:text-ink'}"
							onclick={() => (viewer.layout = layout)}
						>
							{#if layout === "four"}<LayoutGrid size={12} />{/if}
							{LAYOUT_NAMES[layout]}
						</button>
					{/each}
				</div>
				<button class="btn shrink-0" onclick={fit} title="Fit the image (0)"><Maximize2 size={12} /> Fit</button>
			{/if}
			<span class="ml-auto hidden truncate text-2xs text-ink-faint lg:inline">{hint}</span>
		</div>
		<!-- Outside the scrolling part, so it's always in reach. -->
		<button
			class="btn btn-ghost mx-1.5 shrink-0 md:hidden"
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
				title={activeClass ? `Painting ${activeClass.name} (1–9 to change)` : "No class to paint"}
				aria-label={activeClass ? `Active class: ${activeClass.name}` : "No active class"}
				onclick={() => {
					classesOpen = true;
					dockOpen = true;
				}}
			></button>
			<span class="my-1.5 h-px w-6 bg-line"></span>
			<ToolButton icon={Undo2} label="Undo" shortcut="Ctrl+Z" disabled={queue.undoable === 0} onclick={() => queue.undo()} />
			<ToolButton icon={Redo2} label="Redo" shortcut="Ctrl+Shift+Z" disabled={queue.redoable === 0} onclick={() => queue.redo()} />
			<div class="mt-auto">
				<ToolButton icon={Keyboard} label="Keys" shortcut="?" active={viewer.help} onclick={() => (viewer.help = !viewer.help)} />
			</div>
		</div>

		<!-- Document -->
		<div class="relative flex min-w-0 flex-1 flex-col">
			<div class="flex h-7 shrink-0 items-end border-b border-edge bg-chrome px-2">
				<div class="flex h-6 items-center gap-2 rounded-t-sm bg-pasteboard px-3 shadow-[inset_0_1px_0_var(--color-accent)]">
					<span class="font-medium">{title}</span>
					<span class="font-mono text-2xs text-ink-faint">{imageX}×{imageY}×{imageZ} · {manifest.dtype}</span>
				</div>
			</div>
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
						<PlaneView
							{plane}
							{viewer}
							{levels}
							{images}
							{labels}
							prediction={prediction?.store}
							{proposal}
							{segmentation}
							onhover={(p) => (hovered = p)}
							onresize={resized}
							onstroke={stroke}
							onpolygon={() => closePolygon()}
							rois={rois.items}
							onroi={drawRoi}
						/>
					{/each}
					{#if viewer.layout === "four"}
						<div class="flex flex-col justify-center gap-1 bg-pasteboard p-4 font-mono text-2xs text-ink-dim">
							<span><span class="text-axis-x">x</span> {voxel(2)}</span>
							<span><span class="text-axis-y">y</span> {voxel(1)}</span>
							<span><span class="text-axis-z">z</span> {voxel(0)}</span>
						</div>
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
				<div class="grid grid-cols-2 gap-2">
					<label class="label">Black <input class="field font-mono" type="number" step="any" bind:value={viewer.window[0]} /></label>
					<label class="label">White <input class="field font-mono" type="number" step="any" bind:value={viewer.window[1]} /></label>
				</div>
			</Panel>

			<Panel title="Layers">
				<ul class="-mx-2.5 -my-2.5 flex flex-col divide-y divide-edge">
					<li class="flex flex-col gap-1.5 px-2.5 py-2">
						<div class="flex items-center gap-2">
							<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showLabels ? 'Hide' : 'Show'} labels" title="Show or hide (V)" onclick={() => (viewer.showLabels = !viewer.showLabels)}>
								{#if viewer.showLabels}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
							</button>
							<span class="flex-1">Labels</span>
							<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.opacity * 100)}%</span>
						</div>
						<input type="range" min="0" max="1" step="0.05" bind:value={viewer.opacity} aria-label="Label opacity" />
					</li>
					{#if prediction || proposal}
						<li class="flex flex-col gap-1.5 px-2.5 py-2">
							<div class="flex items-center gap-2">
								<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showPrediction ? 'Hide' : 'Show'} prediction" title="Show or hide (M)" onclick={() => (viewer.showPrediction = !viewer.showPrediction)}>
									{#if viewer.showPrediction}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
								</button>
								<span class="flex-1 truncate">
									{prediction ? "Prediction" : "Proposal"} <span class="text-ink-faint">· {(prediction ?? proposal)?.model_name ?? "a model"}</span>
								</span>
								<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.predictionOpacity * 100)}%</span>
							</div>
							{#if prediction && proposal}
								<span class="truncate pl-5.5 text-2xs text-ink-dim">
									Proposal in one ROI <span class="text-ink-faint">· {proposal.model_name ?? "a model"}</span>
								</span>
							{/if}
							<input type="range" min="0" max="1" step="0.05" bind:value={viewer.predictionOpacity} aria-label="Prediction opacity" />
						</li>
					{/if}
					{#if segmentation}
						<li class="flex flex-col gap-1.5 px-2.5 py-2">
							<div class="flex items-center gap-2">
								<button class="text-ink-dim hover:text-ink" aria-label="{viewer.showSegmentation ? 'Hide' : 'Show'} final segmentation" onclick={() => (viewer.showSegmentation = !viewer.showSegmentation)}>
									{#if viewer.showSegmentation}<Eye size={14} />{:else}<EyeOff size={14} />{/if}
								</button>
								<span class="flex-1">Final segmentation</span>
								<span class="font-mono text-2xs text-ink-dim">{Math.round(viewer.segmentationOpacity * 100)}%</span>
							</div>
							<input type="range" min="0" max="1" step="0.05" bind:value={viewer.segmentationOpacity} aria-label="Final segmentation opacity" />
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
				{#if classes.length > 0}
					<ul class="-mx-2.5 -my-1 flex flex-col">
						{#each classes as label, index (label.value)}
							<li>
								<button
									class="flex w-full items-center gap-2 px-2.5 py-1 text-left {viewer.activeClass === label.value ? 'bg-accent-soft text-ink' : 'hover:bg-raised'}"
									onclick={() => (viewer.activeClass = label.value)}
									aria-pressed={viewer.activeClass === label.value}
								>
									<span class="size-3 rounded-[2px] shadow-[0_0_0_1px_black]" style:background={label.color}></span>
									<span class="flex-1">{label.name}</span>
									{#if index < 9}<span class="kbd">{index + 1}</span>{/if}
								</button>
							</li>
						{/each}
					</ul>
				{:else if labels}
					<p class="text-ink-dim">Add label classes in the project settings to start labeling.</p>
				{/if}
			</Panel>

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
					<button
						class="btn"
						disabled={accepting || !!acceptBlocked}
						title={acceptBlocked || `Fill the selected ROI's unlabeled voxels with the ${shownHere.kind} (A)`}
						onclick={() => selectedRoi && acceptPrediction(selectedRoi)}
					>
						<CheckCheck size={13} />
						{accepting ? "Accepting…" : `Accept ${shownHere.kind} here`}
					</button>
				{/if}
				<a href="/p/{project}/rois" class="self-start text-2xs">Open the ROI gallery</a>
			</Panel>
		</aside>
	</div>

	<!-- Status bar -->
	<footer class="flex h-6 shrink-0 items-center gap-4 border-t border-edge bg-chrome px-3 font-mono text-2xs text-ink-dim">
		<span title="Zoom">{zoomPercent}%</span>
		<span>
			<span class="text-axis-x">x</span>{Math.floor(viewer.position[2])}
			<span class="text-axis-y">y</span>{Math.floor(viewer.position[1])}
			<span class="text-axis-z">z</span>{Math.floor(viewer.position[0])}
		</span>
		<span class="hidden sm:inline">{LAYOUT_NAMES[viewer.layout]}</span>
		<span class="ml-auto flex items-center gap-1.5 font-sans {saveState === 'error' ? 'text-danger' : saveState === 'offline' ? 'text-warn' : ''}" role="status">
			{#if saveState === "saved"}<Check size={12} class="text-ok" />{:else if saveState === "saving"}<LoaderCircle size={12} class="animate-spin" />{:else}<CloudOff size={12} />{/if}
			{status}
		</span>
	</footer>
</div>

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
