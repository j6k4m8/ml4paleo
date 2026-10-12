import { describe, expect, it } from "vitest";
import { accessRedirect } from "./access";

describe("public pages", () => {
	it.each(["/", "/about", "/login", "/signup"])("lets signed-out visitors read %s", (path) => {
		expect(accessRedirect(path, "", false, [])).toBeNull();
	});
	it("keeps project pages protected and preserves the return URL", () => {
		expect(accessRedirect("/projects", "?q=skull", false, [])).toBe("/login?next=%2Fprojects%3Fq%3Dskull");
	});
	it("keeps About readable during account setup but gates the workspace", () => {
		expect(accessRedirect("/about", "", true, ["verify_email"])).toBeNull();
		expect(accessRedirect("/projects", "", true, ["set_up_two_factor"])).toBe("/account?next=%2Fprojects");
	});
	it("preserves the administration page through two-factor setup", () => {
		expect(accessRedirect("/admin", "?tab=users", true, ["set_up_two_factor"])).toBe("/account?next=%2Fadmin%3Ftab%3Dusers");
		expect(accessRedirect("/account", "?next=%2Fadmin", true, ["set_up_two_factor"])).toBeNull();
		expect(accessRedirect("/admin", "", true, [])).toBeNull();
	});
});
