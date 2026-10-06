/**
 * Upload a file into a project the way the API expects: start an upload,
 * PUT each part straight to storage with its presigned URL, then complete.
 * Parts storage already has are skipped, so calling this again for the same
 * upload resumes it.
 */

import { api } from "./api";
import type { Upload } from "./types";

const URL_BATCH = 50;

export async function uploadFile(
	projectId: string,
	file: File,
	onProgress: (fraction: number) => void,
	existing?: Upload,
	signal?: AbortSignal,
): Promise<Upload> {
	const base = `/api/projects/${projectId}/uploads`;
	const upload =
		existing ??
		(await api<Upload>(base, { body: { filename: file.name, size: file.size }, signal }));
	const status = await api<Upload>(`${base}/${upload.id}`, { signal });
	const done = new Set(status.stored_parts ?? []);
	const todo = [];
	for (let n = 1; n <= upload.part_count; n++) if (!done.has(n)) todo.push(n);
	let sent = done.size;
	onProgress(sent / upload.part_count);
	for (let i = 0; i < todo.length; i += URL_BATCH) {
		const batch = todo.slice(i, i + URL_BATCH);
		const { urls } = await api<{ urls: Record<string, string> }>(
			`${base}/${upload.id}/part-urls`,
			{ body: { parts: batch }, signal },
		);
		for (const number of batch) {
			const start = (number - 1) * upload.part_size;
			const part = file.slice(start, Math.min(start + upload.part_size, file.size));
			const url = urls[String(number)];
			if (!url) throw new Error(`No URL for part ${number}`);
			// Storage needs only the signed URL, not this site's cookies.
			const response = await fetch(url, { method: "PUT", body: part, signal, credentials: "omit" });
			if (!response.ok) throw new Error(`Part ${number} failed (HTTP ${response.status})`);
			sent += 1;
			onProgress(sent / upload.part_count);
		}
	}
	return api<Upload>(`${base}/${upload.id}/complete`, { method: "POST", signal });
}
