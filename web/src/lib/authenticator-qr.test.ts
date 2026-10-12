import { describe, expect, it, vi } from "vitest";
import jsQR from "jsqr";
import { PNG } from "pngjs";
import { authenticatorQr } from "./authenticator-qr";

describe("authenticator QR codes", () => {
	it("locally encodes a scannable image containing the exact provisioning URI", async () => {
		// Public test key, never a real account's secret.
		const uri = "otpauth://totp/ml4paleo%3Aqr-test?secret=JBSWY3DPEHPK3PXP&issuer=ml4paleo";
		const fetch = vi.spyOn(globalThis, "fetch");
		try {
			const dataUrl = await authenticatorQr(uri);
			expect(dataUrl).toMatch(/^data:image\/png;base64,/);
			const png = PNG.sync.read(Buffer.from(dataUrl.slice(dataUrl.indexOf(",") + 1), "base64"));
			const decoded = jsQR(new Uint8ClampedArray(png.data), png.width, png.height);
			expect(decoded?.data).toBe(uri);
			// Solid white quiet zone, irrespective of the application's theme.
			expect([...png.data.subarray(0, 4)]).toEqual([255, 255, 255, 255]);
			expect(fetch).not.toHaveBeenCalled();
		} finally {
			fetch.mockRestore();
		}
	});
});
