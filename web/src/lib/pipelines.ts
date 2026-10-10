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

export interface ActivityGroup {
	/** Newest run in this consecutive group; its date and model label the row. */
	latest: Pipeline;
	count: number;
}

/** Compact recent activity without hiding errors, progress, or intervening work. */
export function groupActivity(pipelines: Pipeline[]): ActivityGroup[] {
	const groups: ActivityGroup[] = [];
	for (const pipeline of pipelines) {
		const previous = groups.at(-1);
		if (previous && pipeline.kind === previous.latest.kind
			&& pipeline.status === "succeeded" && previous.latest.status === "succeeded"
			&& !pipeline.error && !previous.latest.error) {
			previous.count++;
		} else {
			groups.push({ latest: pipeline, count: 1 });
		}
	}
	return groups;
}

/** A group links to the newest run's model, never an unrelated older model. */
export function activityModelHref(project: string, group: ActivityGroup): string | null {
	const model = group.latest.model_id;
	return model ? `/p/${encodeURIComponent(project)}/models#model-${encodeURIComponent(model)}` : null;
}
