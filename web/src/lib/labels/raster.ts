/**
 * Turn brush strokes and polygons drawn on one plane into voxel masks.
 *
 * Plane coordinates are level-0 voxels along the plane's u and v axes; voxel
 * (i, j) covers [i, i+1) × [j, j+1), so its center is (i + 0.5, j + 0.5).
 * Masks never reach outside the image.
 */

import type { Plane, Vec3 } from "../viewer/tiles";

export class PlaneMask {
	u0 = 0;
	v0 = 0;
	width = 0;
	height = 0;
	bits = new Uint8Array(0);
	count = 0;

	/** `limitU` and `limitV` are the image's size along u and v. */
	constructor(
		private limitU: number,
		private limitV: number,
	) {}

	#grow(u0: number, v0: number, u1: number, v1: number): void {
		if (this.width > 0) {
			u0 = Math.min(u0, this.u0);
			v0 = Math.min(v0, this.v0);
			u1 = Math.max(u1, this.u0 + this.width);
			v1 = Math.max(v1, this.v0 + this.height);
		}
		u0 = Math.max(0, u0);
		v0 = Math.max(0, v0);
		u1 = Math.min(this.limitU, u1);
		v1 = Math.min(this.limitV, v1);
		const width = u1 - u0;
		const height = v1 - v0;
		if (width <= 0 || height <= 0) return;
		if (u0 === this.u0 && v0 === this.v0 && width === this.width && height === this.height) return;
		const bits = new Uint8Array(width * height);
		for (let j = 0; j < this.height; j++) {
			const from = j * this.width;
			bits.set(this.bits.subarray(from, from + this.width), (j + this.v0 - v0) * width + (this.u0 - u0));
		}
		Object.assign(this, { u0, v0, width, height, bits });
	}

	#set(i: number, j: number): void {
		if (i < 0 || j < 0 || i >= this.limitU || j >= this.limitV) return;
		const at = (j - this.v0) * this.width + (i - this.u0);
		if (!this.bits[at]) {
			this.bits[at] = 1;
			this.count++;
		}
	}

	has(i: number, j: number): boolean {
		const a = i - this.u0;
		const b = j - this.v0;
		return a >= 0 && b >= 0 && a < this.width && b < this.height && this.bits[b * this.width + a] === 1;
	}

	/** An ellipse of radii `ru`, `rv` voxels around (cu, cv); at least one voxel. */
	stamp(cu: number, cv: number, ru: number, rv: number): void {
		const i0 = Math.floor(cu - ru);
		const i1 = Math.ceil(cu + ru);
		const j0 = Math.floor(cv - rv);
		const j1 = Math.ceil(cv + rv);
		this.#grow(i0, j0, i1 + 1, j1 + 1);
		if (this.width === 0) return;
		for (let j = j0; j <= j1; j++) {
			const dv = (j + 0.5 - cv) / rv;
			for (let i = i0; i <= i1; i++) {
				const du = (i + 0.5 - cu) / ru;
				if (du * du + dv * dv <= 1) this.#set(i, j);
			}
		}
		this.#set(Math.floor(cu), Math.floor(cv));
	}

	/** Stamps along a segment, close enough together to leave no gaps. */
	line(from: [number, number], to: [number, number], ru: number, rv: number): void {
		const steps = Math.max(1, Math.ceil(Math.hypot((to[0] - from[0]) / ru, (to[1] - from[1]) / rv) * 2));
		for (let k = 1; k <= steps; k++) {
			const t = k / steps;
			this.stamp(from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, ru, rv);
		}
	}

	/**
	 * Fill a polygon, taking voxels whose centers are inside it by the
	 * nonzero winding rule: an outline that loops over itself (a freehand drag
	 * past its start, or round twice) fills everything it goes round, as its
	 * preview shows.
	 */
	polygon(points: [number, number][]): void {
		if (points.length < 3) return;
		const us = points.map((p) => p[0]);
		const vs = points.map((p) => p[1]);
		this.#grow(Math.floor(Math.min(...us)), Math.floor(Math.min(...vs)), Math.ceil(Math.max(...us)) + 1, Math.ceil(Math.max(...vs)) + 1);
		if (this.width === 0) return;
		for (let j = this.v0; j < this.v0 + this.height; j++) {
			const v = j + 0.5;
			// Where the outline crosses this row, and which way it's going.
			const crossings: [number, number][] = [];
			for (let k = 0; k < points.length; k++) {
				const [ua, va] = points[k]!;
				const [ub, vb] = points[(k + 1) % points.length]!;
				if (va <= v !== vb <= v) crossings.push([ua + ((v - va) / (vb - va)) * (ub - ua), vb > va ? 1 : -1]);
			}
			crossings.sort((a, b) => a[0] - b[0]);
			let winding = 0;
			let start = 0;
			for (const [u, direction] of crossings) {
				const before = winding;
				winding += direction;
				if (before === 0) start = u;
				else if (winding === 0) {
					const first = Math.ceil(start - 0.5);
					const last = Math.ceil(u - 0.5) - 1;
					for (let i = first; i <= last; i++) this.#set(i, j);
				}
			}
		}
	}

	/**
	 * The mask as a one-voxel-thick volume at level-0 index `slice` along the
	 * plane's normal: a 0/1 array in (z, y, x) order, its shape, and where it
	 * sits in the image.
	 */
	toVolume(plane: Plane, slice: number): { mask: Uint8Array; shape: Vec3; origin: Vec3 } {
		// Crop to the voxels actually set.
		let [i0, j0, i1, j1] = [Infinity, Infinity, -1, -1];
		for (let j = 0; j < this.height; j++) {
			for (let i = 0; i < this.width; i++) {
				if (!this.bits[j * this.width + i]) continue;
				i0 = Math.min(i0, i);
				i1 = Math.max(i1, i);
				j0 = Math.min(j0, j);
				j1 = Math.max(j1, j);
			}
		}
		const width = Math.max(0, i1 - i0 + 1);
		const height = Math.max(0, j1 - j0 + 1);
		const shape: Vec3 = [1, 1, 1];
		shape[plane.u] = width;
		shape[plane.v] = height;
		const origin: Vec3 = [0, 0, 0];
		origin[plane.normal] = slice;
		origin[plane.u] = this.u0 + (width ? i0 : 0);
		origin[plane.v] = this.v0 + (height ? j0 : 0);
		const mask = new Uint8Array(width * height);
		const c = [0, 0, 0];
		for (let j = 0; j < height; j++) {
			for (let i = 0; i < width; i++) {
				if (!this.bits[(j + j0) * this.width + i + i0]) continue;
				c[plane.u] = i;
				c[plane.v] = j;
				mask[(c[0]! * shape[1] + c[1]!) * shape[2] + c[2]!] = 1;
			}
		}
		return { mask, shape, origin };
	}
}
