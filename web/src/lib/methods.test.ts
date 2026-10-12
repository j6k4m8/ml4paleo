import { describe, expect, it } from "vitest";
import { render } from "svelte/server";
import { METHOD_SECTIONS, composeMethods, methodsResult, type MethodsEntry } from "./methods";
import Methods from "./ui/Methods.svelte";

const entry: MethodsEntry = {
	source: "meshes", artifact_id: "mesh-old", committed_at: "2026-10-12T10:00:00Z", model_id: "old-model",
	paragraph: "", warnings: [], credit: "We used ml4paleo (citation).", lineage_warnings: ["Uses the saved result."],
	sections: [
		{ key: "training", text: "We trained 100 trees.", warnings: [] },
		{ key: "performance", text: "Mean Dice was 0.946.", warnings: ["Internal estimate."] },
		{ key: "prediction", text: "We predicted in chunks.", warnings: [] },
		{ key: "segmentation", text: "We combined the labels.", warnings: [] },
		{ key: "meshes", text: "We generated meshes with zmesh.", warnings: [] },
	],
};

describe("selectable Methods paragraph", () => {
	it("combines selected parts in methods order with a single citation", () => {
		const result = composeMethods(entry, ["meshes", "training", "performance"]);
		expect(result.paragraph).toBe("We used ml4paleo (citation). We trained 100 trees. Mean Dice was 0.946. We generated meshes with zmesh.");
		expect(result.warnings).toEqual(["Uses the saved result.", "Internal estimate."]);
	});
	it("omits both performance text and its warnings when unchecked", () => {
		const result = composeMethods(entry, ["training"]);
		expect(result.paragraph).not.toContain("Dice");
		expect(result.warnings).not.toContain("Internal estimate.");
	});
	it("does not offer a citation-only paragraph when nothing is selected", () => {
		expect(composeMethods(entry, []).paragraph).toBe("");
		expect(composeMethods(undefined, ["training"]).paragraph).toBe("");
	});
	it("keeps one pinned lineage even when a newer prediction exists", () => {
		const prediction = { ...entry, source: "prediction" as const, artifact_id: "new-prediction", model_id: "new-model" };
		expect(methodsResult([entry, prediction])).toBe(entry);
		expect(methodsResult([prediction])).toBe(prediction);
	});
	it("renders collapsed by default, with section checkboxes and no selector", () => {
		const { body } = render(Methods, { props: { entries: [entry], pid: "test" } });
		expect(body).toContain("<details");
		expect(body).not.toMatch(/<details[^>]*\bopen(?:[ =>])/);
		expect(body).not.toContain("<select");
		expect(body.match(/type="checkbox"/g)).toHaveLength(5);
		for (const [, label] of METHOD_SECTIONS) expect(body).toContain(label);
		expect(body).toContain(composeMethods(entry, METHOD_SECTIONS.map(([key]) => key)).paragraph);
	});
	it("does not invent details for unavailable sections", () => {
		const prediction = { ...entry, source: "prediction" as const, sections: entry.sections.slice(0, 3) };
		const { body } = render(Methods, { props: { entries: [prediction], pid: "test" } });
		expect(body.match(/disabled/g)).toHaveLength(2);
		expect(composeMethods(prediction, ["meshes"]).paragraph).toBe("");
	});
});
