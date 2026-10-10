import { describe, expect, it, vi } from "vitest";
import { PLANES, type Plane } from "../viewer/tiles";
import { acceptGestureGroups, type GestureSource } from "./accept-gesture";
import type { Box } from "./accept";
import { decodeDelta } from "./deltas";
import { OpQueue, type GroupedEdit, type OpOut } from "./opqueue.svelte";
import { PlaneMask } from "./raster";

const image: Box = [0, 0, 0, 128, 128, 128];
const reader = (value: (z: number, y: number, x: number) => number) => async (box: Box) => {
	const out: number[] = [];
	for (let z = box[0]; z < box[3]; z++) for (let y = box[1]; y < box[4]; y++) for (let x = box[2]; x < box[5]; x++) out.push(value(z, y, x));
	return Uint8Array.from(out);
};
const source = (read: GestureSource["read"], box = image, artifact = "prediction"): GestureSource => ({ artifact, box, read });
const labels = reader(() => 0);
const allClasses = new Set([2, 3, 4]);
const tool = { name: "accept-brush" };

function written(groups: GroupedEdit[]): Map<string, number> {
	const result = new Map<string, number>();
	for (const delta of groups.flatMap((g) => g.parts.flat())) {
		const { mask, written } = decodeDelta(delta);
		let i = 0;
		for (let z = delta.box[0]; z < delta.box[3]; z++) {
			for (let y = delta.box[1]; y < delta.box[4]; y++) {
				for (let x = delta.box[2]; x < delta.box[5]; x++, i++) {
					if (mask[i]) result.set([z + delta.key[0] * 64, y + delta.key[1] * 64, x + delta.key[2] * 64].join("/"), typeof written === "number" ? written : written[i]!);
				}
			}
		}
	}
	return result;
}

describe("Accept brush and polygon", () => {
	it.each(Object.values(PLANES))("accepts exactly a brush's selected voxels on $name", async (plane: Plane) => {
		const mask = new PlaneMask(128, 128);
		mask.stamp(5.5, 7.5, 1.6, 2.2);
		mask.line([5.5, 7.5], [12.5, 10.5], 1.6, 2.2);
		const result = await acceptGestureGroups(plane, 11, mask, [source(reader(() => 2))], labels, {}, allClasses, tool);
		const got = written(result.groups);
		expect(got.size).toBe(mask.count);
		expect(result.count).toBe(mask.count);
		for (const key of got.keys()) {
			const point = key.split("/").map(Number);
			expect(point[plane.normal]).toBe(11);
			expect(mask.has(point[plane.u]!, point[plane.v]!)).toBe(true);
		}
	});

	it.each(Object.values(PLANES))("accepts the polygon interior, not its bounding box, on $name", async (plane: Plane) => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[2, 2], [8, 2], [2, 8]]);
		const result = await acceptGestureGroups(plane, 4, mask, [source(reader(() => 3))], labels, {}, allClasses, { name: "accept-polygon" });
		expect(result.count).toBe(mask.count);
		expect(result.count).toBeLessThan(mask.width * mask.height);
		for (const [key, value] of written(result.groups)) {
			const point = key.split("/").map(Number);
			expect(point[plane.normal]).toBe(4);
			expect(mask.has(point[plane.u]!, point[plane.v]!)).toBe(true);
			expect(value).toBe(3);
		}
	});

	it("keeps each predicted class and skips saved/declined labels, Background, hidden, transparent and unknown classes", async () => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[0, 0], [12, 0], [12, 1], [0, 1]]);
		const predicted = [0, 1, 2, 3, 4, 5, 2, 3, 2, 2, 255, 7];
		const saved =     [0, 0, 0, 0, 0, 0, 3, 1, 255, 0, 0, 0];
		const result = await acceptGestureGroups(PLANES.xy, 0, mask,
			[source(reader((_z, _y, x) => predicted[x] ?? 0))], reader((_z, _y, x) => saved[x] ?? 0),
			{ 4: { visible: false }, 5: { opacity: 0 } }, new Set([2, 3, 4, 5]), tool);
		expect(written(result.groups)).toEqual(new Map([["0/0/2", 2], ["0/0/3", 3], ["0/0/9", 2]]));
		expect(result.count).toBe(3);
		expect(result.groups.flatMap((g) => g.parts.flat()).every((d) => d.only_if === "unlabeled")).toBe(true);
	});

	it("keeps distinct chunk artifacts and groups their classes in one undo step", async () => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[62, 1], [67, 1], [67, 3], [62, 3]]);
		const sources = [
			source(reader((_z, y) => y === 1 ? 2 : 3), [0, 0, 0, 64, 64, 64], "left"),
			source(reader(() => 4), [0, 0, 64, 64, 64, 128], "right"),
		];
		const result = await acceptGestureGroups(PLANES.xy, 2, mask, sources, labels, {}, allClasses, tool);
		expect(result.count).toBe(10);
		expect(result.groups.map((g) => g.options?.accept?.prediction)).toEqual(["left", "right"]);
		for (const group of result.groups) {
			expect(group.parts.flat().every((d) => d.key[2] === (group.options!.accept!.prediction === "left" ? 0 : 1))).toBe(true);
		}
		// Pause sending: the full multi-artifact gesture is still a single local undo group.
		const send = vi.fn(async () => ({ seq: 1, chunks: [] } as unknown as OpOut));
		const queue = new OpQueue("project", null, send);
		queue.stop();
		const ops = queue.editTogether(result.groups);
		expect(ops).toHaveLength(3);
		expect(queue.undoable).toBe(1);
		expect(queue.undo()).toBe(true);
		expect(queue.pending).toBe(0);
		expect(queue.redoable).toBe(1);
		expect(queue.redo()).toBe(true);
		expect(queue.pending).toBe(3);
		expect(queue.undoable).toBe(1);
		expect(send).not.toHaveBeenCalled();
	});

	it.each(["accept-brush", "accept-polygon"])("filters predicted classes, not saved classes, for %s", async (name) => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[0, 0], [10, 0], [10, 1], [0, 1]]);
		const predicted = [2, 3, 4, 3, 4, 2, 3, 4, 1, 3];
		const saved =     [0, 0, 0, 2, 3, 3, 255, 1, 0, 0];
		const sources = [source(reader((_z, _y, x) => predicted[x]!))];
		const readLabels = reader((_z, _y, x) => saved[x]!);
		const selected = await acceptGestureGroups(PLANES.xy, 0, mask, sources, readLabels, {}, new Set([3, 4]), { name });
		expect(written(selected.groups)).toEqual(new Map([["0/0/1", 3], ["0/0/2", 4], ["0/0/9", 3]]));
		expect(selected.count).toBe(3);
		expect(selected.groups.flatMap((g) => g.parts.flat()).every((d) => d.only_if === "unlabeled")).toBe(true);
		const all = await acceptGestureGroups(PLANES.xy, 0, mask, sources, readLabels, {}, allClasses, { name });
		expect(all.count).toBe(4);
		expect(written(all.groups).get("0/0/0")).toBe(2);
	});

	it("makes no edits when no predicted classes match the Accept filter", async () => {
		const mask = new PlaneMask(128, 128);
		mask.stamp(4, 4, 2, 2);
		expect(await acceptGestureGroups(PLANES.xy, 0, mask, [source(reader(() => 2))], labels, {}, new Set([3]), tool))
			.toEqual({ groups: [], count: 0 });
	});

	it("does not accept unready gaps in a Live prediction", async () => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[62, 1], [67, 1], [67, 3], [62, 3]]);
		const result = await acceptGestureGroups(PLANES.xy, 2, mask,
			[source(reader(() => 2), [0, 0, 0, 64, 64, 64])], labels, {}, allClasses, tool);
		expect(result.count).toBe(4);
		expect([...written(result.groups).keys()].every((key) => Number(key.split("/")[2]) < 64)).toBe(true);
	});

	it("clips gestures at the visible slice even if a drag extends outside the canvas", async () => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[0, 0], [20, 0], [20, 20], [0, 20]]);
		const read = vi.fn(reader(() => 2));
		const view: Box = [4, 5, 6, 5, 10, 11];
		const result = await acceptGestureGroups(PLANES.xy, 4, mask, [source(read, view)], labels, {}, allClasses, tool);
		expect(result.count).toBe(25);
		expect(read).toHaveBeenCalledWith(view);
	});

	it("returns no edits for an empty or fully masked gesture", async () => {
		const mask = new PlaneMask(128, 128);
		expect(await acceptGestureGroups(PLANES.xy, 0, mask, [], labels, {}, allClasses, tool)).toEqual({ groups: [], count: 0 });
		mask.stamp(4, 4, 2, 2);
		expect(await acceptGestureGroups(PLANES.xy, 0, mask, [source(reader(() => 2))], labels, { 2: { visible: false } }, allClasses, tool)).toEqual({ groups: [], count: 0 });
	});

	it("refuses missing or inconsistent data before returning any edits", async () => {
		const mask = new PlaneMask(128, 128);
		mask.polygon([[62, 1], [67, 1], [67, 3], [62, 3]]);
		await expect(acceptGestureGroups(PLANES.xy, 2, mask, [
			source(reader(() => 2), [0, 0, 0, 64, 64, 64]),
			source(async () => { throw new Error("not loaded"); }, [0, 0, 64, 64, 64, 128]),
		], labels, {}, allClasses, tool)).rejects.toThrow("not loaded");
		await expect(acceptGestureGroups(PLANES.xy, 2, mask, [source(async () => new Uint8Array(0))],
			labels, {}, allClasses, tool)).rejects.toThrow("Incomplete");
	});

	it("bounds very large gestures before reading prediction data", async () => {
		const mask = new PlaneMask(10000, 10000);
		mask.stamp(0, 0, 1, 1);
		// No enormous fixture allocation: these bounds alone must reject it.
		mask.width = mask.height = 10000;
		const read = vi.fn(labels);
		await expect(acceptGestureGroups(PLANES.xy, 0, mask, [source(read)], labels, {}, allClasses, tool)).rejects.toThrow("too big");
		expect(read).not.toHaveBeenCalled();
	});
});
