import { ChunkMeshCache, MeshDetailError } from "./chunk-mesh";
import { suggestionChunks } from "./minimap";
import type { MeshRequest, MeshReply } from "./minimap-meshing";
import { compressedBody } from "../compression";

const cache = new ChunkMeshCache();

self.onmessage = async (event: MessageEvent<MeshRequest>) => {
	const { id, layer, input, sources, ids, project, headers } = event.data;
	try {
		const chunks = layer === "suggestions"
			? suggestionChunks(ids ?? [], sources ?? [], new Map(input.chunks.map(({ id, chunk }) => [id, chunk])), input.colors)
			: input.chunks;
		const { data, downsample } = await cache.build(layer, { ...input, chunks }, async (raw, key, factor) => {
			const wire = await compressedBody(raw);
			let response: Response | undefined;
			for (let attempt = 0; attempt < 4; attempt++) {
				response = await fetch(`/api/projects/${project}/meshes/preview?chunk=${key}&downsample=${factor}`, { method: "POST", headers: { ...headers, ...wire.headers, "Content-Type": "application/octet-stream" }, body: wire.body, credentials: "same-origin", signal: AbortSignal.timeout(30000) });
				if (response.status !== 503) break;
				await response.body?.cancel();
				await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
			}
			if (!response?.ok) {
				const message = await response?.json().catch(() => null);
				if (response?.status === 422 && message?.detail?.code === "mesh_detail") throw new MeshDetailError();
				throw new Error(typeof message?.detail === "string" ? message.detail : "Couldn't load the 3D surface.");
			}
			return await response.arrayBuffer();
		});
		(self as unknown as Worker).postMessage({ id, layer, data, downsample } satisfies MeshReply, [data.buffer]);
	} catch (error) {
		(self as unknown as Worker).postMessage({ id, layer, error: error instanceof Error ? error.message : "Couldn't build the 3D surface." } satisfies MeshReply);
	}
};
