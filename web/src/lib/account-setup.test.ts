import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "svelte/server";

const state = vi.hoisted(() => ({
	current: {
		user: { username: "test", email: "test@example.test", email_verified: true, two_factor_enabled: false },
		required_steps: [] as string[],
	},
}));
vi.mock("#lib/session.svelte.ts", () => ({ session: state }));
vi.mock("$app/state", () => ({ page: { url: new URL("http://localhost/account?next=%2Fadmin") } }));
vi.mock("$app/navigation", () => ({ goto: vi.fn() }));
import Account from "../routes/account/+page.svelte";

describe("required account setup", () => {
	beforeEach(() => {
		state.current.required_steps = [];
		state.current.user.two_factor_enabled = false;
	});
	it("explains the admin requirement and puts two-factor setup before unrelated account settings", () => {
		state.current.required_steps = ["set_up_two_factor"];
		const { body } = render(Account);
		expect(body).toContain("Administration requires two-factor sign-in.");
		expect(body.indexOf("Show QR code")).toBeLessThan(body.indexOf("test@example.test"));
		expect(body).not.toContain("Finish these steps");
		expect(body).not.toContain("Current password");
		expect(body).not.toContain("New password");
	});
	it("shows the password fields only when a password change is actually required", () => {
		state.current.required_steps = ["change_password", "set_up_two_factor"];
		const { body } = render(Account);
		expect(body).toContain("Change your temporary password before continuing.");
		expect(body).toContain("Current password");
		expect(body).toContain("Show QR code");
	});
	it("does not imply two-factor setup is required for a password-only reset", () => {
		state.current.required_steps = ["change_password"];
		const { body } = render(Account);
		expect(body).not.toContain("Administration requires");
		expect(body).not.toContain("Show QR code");
		expect(body).toContain("Current password");
	});
	it("shows a clear enabled state after setup", () => {
		state.current.user.two_factor_enabled = true;
		const { body } = render(Account);
		expect(body).toContain("Two-factor sign-in is on.");
		expect(body).not.toContain("Show QR code");
	});
});
