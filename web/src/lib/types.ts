/** Shapes of the API's answers, as the app uses them. */

export interface User {
	id: string;
	username: string;
	email: string | null;
	is_admin: boolean;
	status: string;
}

export interface Session {
	user: User;
	csrf_token: string;
	required_steps: string[];
}

export interface Project {
	id: string;
	name: string;
	owner: string;
	created_at: string;
}

export interface Upload {
	id: string;
	filename: string;
	size: number;
	part_size: number;
	part_count: number;
	state: string;
	stored_parts: number[] | null;
}

export interface Pipeline {
	id: string;
	kind: string;
	status: "waiting" | "running" | "succeeded" | "failed" | "cancelled";
	progress: number;
	jobs: number;
	created_at: string;
	error: string | null;
}

export interface ImageManifest {
	shape_czyx: [number, number, number, number];
	dtype: string;
	levels: number;
	voxel_size_zyx: [number, number, number] | null;
	unit: string | null;
	window: [number, number];
	min: number;
	max: number;
}

export interface ProjectImage {
	artifact_id: string;
	manifest: ImageManifest;
	zarr_url: string;
	neuroglancer_url: string | null;
}
