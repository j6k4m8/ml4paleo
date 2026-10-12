import QRCode from "qrcode";

/** Encode the server's provisioning URI locally; never send setup keys to a QR service. */
export function authenticatorQr(uri: string): Promise<string> {
	return QRCode.toDataURL(uri, {
		errorCorrectionLevel: "M",
		margin: 4,
		scale: 6,
		color: { dark: "#000000", light: "#ffffff" },
	});
}
