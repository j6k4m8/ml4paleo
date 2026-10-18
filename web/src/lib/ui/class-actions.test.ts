import { describe, expect, it } from "vitest";
import { render } from "svelte/server";
import ClassActionDialog from "./ClassActionDialog.svelte";
import ClassActions from "./ClassActions.svelte";

const label = { value: 2, name: "Bone", color: "#ff0000" };
const props = { label, mode: "reset" as const, busy: false, error: "", onconfirm: () => {}, oncancel: () => {} };

describe("irreversible class actions", () => {
	it("explains the full-scan scope, permanence, and exact-name confirmation", () => {
		const { body } = render(ClassActionDialog, { props });
		expect(body).toContain("across the entire scan");
		expect(body).toContain("The class will remain available.");
		expect(body).toContain("This cannot be undone.");
		expect(body).toContain("Type “Bone” to confirm");
		expect(body).toMatch(/<button[^>]*disabled[^>]*>\s*Reset this class/);
	});
	it.each(["", "bone", "Bone ", "Other"])("refuses the wrong confirmation %j", (confirmation) => {
		const { body } = render(ClassActionDialog, { props: { ...props, confirmation } });
		expect(body).toMatch(/<button[^>]*disabled[^>]*>\s*Reset this class/);
	});
	it("enables only an exact match and prevents a duplicate while busy", () => {
		const { body } = render(ClassActionDialog, { props: { ...props, confirmation: "Bone" } });
		expect(body).not.toMatch(/<button[^>]*\sdisabled(?:\s|>)/);
		const busy = render(ClassActionDialog, { props: { ...props, confirmation: "Bone", busy: true } }).body;
		expect(busy).toMatch(/<button[^>]*disabled[^>]*>\s*Resetting/);
	});
	it("makes deleting the class distinct from resetting it, and shows errors", () => {
		const { body } = render(ClassActionDialog, { props: { ...props, mode: "delete", error: "Type the class name exactly to confirm." } });
		expect(body).toContain("The class will also be removed from the project.");
		expect(body).toContain("Delete this class");
		expect(body).toContain('role="alert"');
		expect(body).toContain("Type the class name exactly to confirm.");
	});
	it("offers both actions, but background can only be reset", () => {
		const { body } = render(ClassActions, { props: { pid: "project", label, onchanged: () => {} } });
		expect(body).toContain("Reset this class (wipe all paint)");
		expect(body).toContain("Delete this class (remove this label from the project)");
		expect(body).not.toMatch(/<button[^>]*\sdisabled(?:\s|>)/);
		const background = render(ClassActions, { props: { pid: "project", label: { ...label, value: 1, name: "Background" }, onchanged: () => {} } }).body;
		expect(background).toMatch(/<button[^>]*disabled[^>]*>Delete this class/);
	});
});
