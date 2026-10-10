import type { Box } from "../rois.svelte";
import type { Chunk } from "./chunks";
import type { SurfaceInput } from "./minimap-mesh";
import { csrfHeaders } from "../api";

export type MeshLayer = "labels" | "suggestions";
export interface MeshRequest {
	id: number;
	layer: MeshLayer;
	input: SurfaceInput;
	sources?: readonly { box: Box; chunks: ReadonlyMap<string, Chunk> }[];
	ids?: readonly string[];
	project: string;
	headers: Record<string, string>;
}
export interface MeshReply { id: number; layer: MeshLayer; data?: Float32Array; error?: string; downsample?: number }

/** One active build and one latest request per layer. Superseded results cannot publish. */
export class MinimapMesher {
	#worker: Worker;
	#next = 0;
	#active = false;
	#versions = new Map<MeshLayer, number>();
	#pending = new Map<MeshLayer, MeshRequest>();
	constructor(private project: string, private receive: (reply: MeshReply) => void, makeWorker = () => new Worker(new URL("./minimap-mesh.worker.ts", import.meta.url), { type: "module" })) {
		this.#worker = makeWorker();
		this.#worker.onmessage = (event: MessageEvent<MeshReply>) => {
			this.#active = false;
			if (this.#versions.get(event.data.layer) === event.data.id) this.receive(event.data);
			this.send();
		};
		this.#worker.onerror = (event) => {
			event.preventDefault();
			for (const [layer, id] of this.#versions) this.receive({ layer, id, error: "Couldn't build the 3D surface. Reload to retry." });
			this.destroy();
		};
	}
	request(layer: MeshLayer, input: SurfaceInput, sources?: MeshRequest["sources"], ids?: readonly string[]): void {
		const id = ++this.#next;
		this.#versions.set(layer, id);
		this.#pending.set(layer, { id, layer, input, sources, ids, project: this.project, headers: csrfHeaders() });
		this.send();
	}
	cancel(layer: MeshLayer): void { this.#versions.delete(layer); this.#pending.delete(layer); }
	private send(): void {
		if (this.#active) return;
		const next = this.#pending.values().next().value as MeshRequest | undefined;
		if (!next) return;
		this.#pending.delete(next.layer);
		this.#active = true;
		// Structured clone leaves the shared chunk cache intact; output is transferred back.
		this.#worker.postMessage(next);
	}
	destroy(): void { this.#worker.terminate(); this.#versions.clear(); this.#pending.clear(); this.#active = true; }
}
