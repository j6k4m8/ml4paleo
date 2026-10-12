export const METHOD_SECTIONS = [
	["training", "Model training"],
	["performance", "Model performance"],
	["prediction", "Prediction"],
	["segmentation", "Final segmentation"],
	["meshes", "Meshing"],
] as const;

export type MethodSectionKey = (typeof METHOD_SECTIONS)[number][0];
export interface MethodsEntry {
	source: "prediction" | "segmentation" | "meshes";
	artifact_id: string;
	committed_at: string;
	model_id: string | null;
	paragraph: string;
	warnings: string[];
	credit: string;
	sections: { key: MethodSectionKey; text: string; warnings: string[] }[];
	lineage_warnings: string[];
}

/** Anchor all selected parts to one result, never mix different model generations. */
export function methodsResult(entries: MethodsEntry[]): MethodsEntry | undefined {
	return entries.find((entry) => entry.source === "meshes")
		?? entries.find((entry) => entry.source === "segmentation")
		?? entries.find((entry) => entry.source === "prediction");
}

export function composeMethods(entry: MethodsEntry | undefined, selected: readonly MethodSectionKey[]) {
	const sections = METHOD_SECTIONS.filter(([key]) => selected.includes(key))
		.map(([key]) => entry?.sections.find((section) => section.key === key))
		.filter((section) => section !== undefined);
	const parts = sections.map((section) => section.text.trim()).filter(Boolean);
	return {
		paragraph: parts.length ? [entry?.credit, ...parts].filter(Boolean).join(" ") : "",
		warnings: [...new Set([...(entry?.lineage_warnings ?? []), ...sections.flatMap((section) => section.warnings)])],
	};
}
