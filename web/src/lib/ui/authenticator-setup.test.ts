import { describe, expect, it } from "vitest";
import { render } from "svelte/server";
import AuthenticatorSetup from "./AuthenticatorSetup.svelte";

const props = {
	setup: null, qr: "", qrError: false, busy: false, error: "",
	onstart: () => {}, onconfirm: () => {},
};
const setup = { secret: "PUBLIC_TEST_KEY", otpauth_uri: "otpauth://totp/ml4paleo?secret=PUBLIC_TEST_KEY" };

describe("phone-friendly two-factor setup", () => {
	it("starts with a clear QR action and explains which phone app to use", () => {
		const { body } = render(AuthenticatorSetup, { props });
		expect(body).toContain("Show QR code");
		expect(body).toContain("authenticator app on your phone");
		expect(body).not.toContain("Current password");
	});
	it("shows the QR, six-digit field, and collapsed manual fallback", () => {
		const { body } = render(AuthenticatorSetup, { props: { ...props, setup, qr: "data:image/png;base64,test" } });
		expect(body).toContain('src="data:image/png;base64,test"');
		expect(body).toContain("Scan QR code");
		expect(body).toContain("6-digit code from your authenticator app");
		expect(body).toContain('autocomplete="one-time-code"');
		expect(body).toContain('pattern="[0-9]{6}"');
		expect(body).toContain('href="otpauth://totp/ml4paleo?secret=PUBLIC_TEST_KEY"');
		expect(body).toContain("<details>");
		expect(body).not.toContain("<details open");
	});
	it("opens the manual fallback if QR generation fails", () => {
		const { body } = render(AuthenticatorSetup, { props: { ...props, setup, qrError: true } });
		expect(body).toContain("<details open");
		expect(body).toContain("PUBLIC_TEST_KEY");
		expect(body).toContain("The QR code couldn’t load");
	});
	it("disables confirmation while it is being checked and keeps errors visible", () => {
		const { body } = render(AuthenticatorSetup, { props: { ...props, setup, busy: true, error: "Try the next code." } });
		expect(body).toContain("disabled");
		expect(body).toContain("Checking…");
		expect(body).toContain('role="alert"');
		expect(body).toContain("Try the next code.");
	});
});
