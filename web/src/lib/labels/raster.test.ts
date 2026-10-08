import { describe, expect, it } from "vitest";
import { PLANES } from "../viewer/tiles";
import { PlaneMask } from "./raster";

function cells(mask: PlaneMask): string[] {
	const out: string[] = [];
	for (let j = mask.v0; j < mask.v0 + mask.height; j++) {
		for (let i = mask.u0; i < mask.u0 + mask.width; i++) if (mask.has(i, j)) out.push(`${i},${j}`);
	}
	return out;
}

describe("PlaneMask", () => {
	it("stamps discs by voxel centers, and at least one voxel", () => {
		const disc = new PlaneMask(100, 100);
		disc.stamp(10.5, 10.5, 1.5, 1.5);
		expect(disc.count).toBe(9);
		const dot = new PlaneMask(100, 100);
		dot.stamp(4.2, 7.9, 0.1, 0.1);
		expect(cells(dot)).toEqual(["4,7"]);
	});

	it("stretches discs for thick voxels", () => {
		const mask = new PlaneMask(100, 100);
		mask.stamp(10.5, 10.5, 4, 1);
		expect(mask.width).toBeGreaterThan(mask.height);
		expect(mask.has(13, 10) && !mask.has(10, 12)).toBe(true);
	});

	it("draws lines without gaps", () => {
		const mask = new PlaneMask(100, 100);
		mask.line([2, 2], [40, 2], 0.6, 0.6);
		for (let i = 2; i < 40; i++) expect(mask.has(i, 1) || mask.has(i, 2)).toBe(true);
	});

	it("fills polygons by voxel centers", () => {
		const mask = new PlaneMask(100, 100);
		mask.polygon([
			[2, 2],
			[6, 2],
			[6, 5],
			[2, 5],
		]);
		expect(mask.count).toBe(12);
		expect(mask.has(2, 2) && mask.has(5, 4) && !mask.has(6, 2) && !mask.has(2, 5)).toBe(true);
	});

	it("fills outlines that loop over themselves by winding, as their previews show", () => {
		// A freehand circle that goes on past its start fills the whole disc.
		const turns = 1.3;
		const around = Array.from({ length: 200 }, (_, i): [number, number] => {
			const angle = (2 * Math.PI * turns * i) / 199;
			return [50 + 20 * Math.cos(angle), 50 + 20 * Math.sin(angle)];
		});
		const past = new PlaneMask(100, 100);
		past.polygon(around);
		expect(past.count).toBeGreaterThan(Math.PI * 400 * 0.97);
		expect(past.count).toBeLessThan(Math.PI * 400 * 1.03);
		// Round twice fills as round once.
		const once = new PlaneMask(100, 100);
		once.polygon([
			[10, 10],
			[30, 10],
			[30, 30],
			[10, 30],
		]);
		const twice = new PlaneMask(100, 100);
		twice.polygon([
			[10, 10],
			[30, 10],
			[30, 30],
			[10, 30],
			[10, 10],
			[30, 10],
			[30, 30],
			[10, 30],
		]);
		expect(cells(twice)).toEqual(cells(once));
		// A star fills its middle; a figure eight fills both loops and nothing between.
		const star = new PlaneMask(100, 100);
		star.polygon(
			[0, 2, 4, 1, 3].map((k): [number, number] => {
				const angle = -Math.PI / 2 + (2 * Math.PI * k) / 5;
				return [50 + 30 * Math.cos(angle), 50 + 30 * Math.sin(angle)];
			}),
		);
		expect(star.has(50, 50) && star.has(50, 25) && !star.has(30, 25)).toBe(true);
		const eight = new PlaneMask(100, 100);
		eight.polygon([
			[10, 10],
			[30, 30],
			[30, 10],
			[10, 30],
		]);
		expect(eight.has(12, 20) && eight.has(27, 20) && !eight.has(20, 12) && !eight.has(20, 27)).toBe(true);
	});

	it("stays inside the image", () => {
		const mask = new PlaneMask(10, 10);
		mask.stamp(0, 0, 3, 3);
		mask.polygon([
			[8, 8],
			[20, 8],
			[20, 20],
		]);
		expect(mask.u0).toBe(0);
		expect(mask.u0 + mask.width).toBeLessThanOrEqual(10);
		expect(mask.v0 + mask.height).toBeLessThanOrEqual(10);
		expect(cells(mask).every((c) => c.split(",").every((n) => Number(n) >= 0 && Number(n) < 10))).toBe(true);
	});

	it("places the mask in the volume for each plane", () => {
		const mask = new PlaneMask(100, 100);
		mask.stamp(3.5, 7.5, 0.1, 0.1); // voxel u=3, v=7
		mask.stamp(4.5, 7.5, 0.1, 0.1); // u=4
		const xy = mask.toVolume(PLANES.xy, 20);
		expect(xy).toEqual({ mask: new Uint8Array([1, 1]), shape: [1, 1, 2], origin: [20, 7, 3] });
		const xz = mask.toVolume(PLANES.xz, 20);
		expect(xz).toEqual({ mask: new Uint8Array([1, 1]), shape: [1, 1, 2], origin: [7, 20, 3] });
		// YZ: u is z, v is y.
		const yz = mask.toVolume(PLANES.yz, 20);
		expect(yz).toEqual({ mask: new Uint8Array([1, 1]), shape: [2, 1, 1], origin: [3, 7, 20] });
	});
});
