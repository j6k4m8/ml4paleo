/** Compress large binary requests off the UI thread (called by the mesh worker).
 * Responses use HTTP gzip, which fetch transparently decodes. Stored Zarr
 * chunks already have their own compression and must not pass through here.
 */
export async function compressedBody(raw: ArrayBuffer): Promise<{ body: ArrayBuffer; headers: Record<string, string> }> {
	if (raw.byteLength >= 1024 && typeof CompressionStream !== "undefined") {
		try {
			const stream = new Blob([raw]).stream().pipeThrough(new CompressionStream("gzip"));
			const body = await new Response(stream).arrayBuffer();
			if (body.byteLength < raw.byteLength) return { body, headers: { "Content-Encoding": "gzip" } };
		} catch {
			// Older browsers may expose CompressionStream without gzip support.
		}
	}
	return { body: raw, headers: {} };
}
