import { expect, it, vi } from "vitest";
import { MinimapMesher, type MeshReply, type MeshRequest } from "./minimap-meshing";
import type { SurfaceInput } from "./minimap-mesh";

it("runs one build at a time, coalesces each layer, and discards stale/canceled replies", () => {
	const worker = { postMessage: vi.fn(), terminate: vi.fn(), onmessage: (_: MessageEvent<MeshReply>) => {}, onerror: (_: ErrorEvent) => {} };
	const receive = vi.fn(), mesher = new MinimapMesher("project", receive, () => worker as unknown as Worker);
	const input: SurfaceInput = { level: { index: 0, path: "class", shape: [1, 1, 1], scale: [1, 1, 1] }, chunks: [], colors: new Map(), shape: [1, 1, 1], aspect: [1, 1, 1] };
	const respond = (request: MeshRequest) => worker.onmessage({ data: { id: request.id, layer: request.layer, data: new Float32Array() } } as MessageEvent<MeshReply>);
	mesher.request("labels", input);
	mesher.request("labels", input); mesher.request("labels", input);
	mesher.request("suggestions", input);
	expect(worker.postMessage).toHaveBeenCalledTimes(1);
	respond(worker.postMessage.mock.calls[0]![0]);
	expect(receive).not.toHaveBeenCalled();
	expect(worker.postMessage.mock.calls[1]![0].id).toBe(3);
	respond(worker.postMessage.mock.calls[1]![0]);
	expect(receive).toHaveBeenCalledTimes(1);
	mesher.cancel("suggestions");
	respond(worker.postMessage.mock.calls[2]![0]);
	expect(receive).toHaveBeenCalledTimes(1);
	mesher.destroy();
	expect(worker.terminate).toHaveBeenCalledOnce();
});
