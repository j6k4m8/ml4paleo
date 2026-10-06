/** Following a project's pipelines. */

import type { Pipeline } from "./types";

/** Whether a pipeline is still waiting or running. */
export function unfinished(pipeline: Pipeline): boolean {
	return pipeline.status === "waiting" || pipeline.status === "running";
}

/** Each model's latest prediction, from a project's pipelines (newest first). */
export function latestPredictions(pipelines: Pipeline[]): Record<string, Pipeline> {
	const latest: Record<string, Pipeline> = {};
	for (const pipeline of pipelines) {
		const model = pipeline.model_id;
		if (pipeline.kind === "prediction" && model && !(model in latest)) latest[model] = pipeline;
	}
	return latest;
}
