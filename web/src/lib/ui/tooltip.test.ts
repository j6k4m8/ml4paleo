import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tooltip, tooltipPosition } from "./tooltip";

class ElementStub extends EventTarget {
	id = "";
	className = "";
	hidden = false;
	textContent = "";
	style: Record<string, string> = {};
	attributes = new Map<string, string>();
	constructor(readonly ownerDocument: DocumentStub) { super(); }
	getAttribute(key: string) { return this.attributes.get(key) ?? null; }
	setAttribute(key: string, value: string) { this.attributes.set(key, value); }
	removeAttribute(key: string) { this.attributes.delete(key); }
	getBoundingClientRect() { return { left: 20, top: 20, bottom: 52, width: 100, height: 32 }; }
	remove() { this.ownerDocument.children.delete(this); }
}
class DocumentStub extends EventTarget {
	children = new Set<ElementStub>();
	body = { append: (node: ElementStub) => this.children.add(node) };
	defaultView = Object.assign(new EventTarget(), { innerWidth: 800, innerHeight: 600 });
	createElement() { return new ElementStub(this); }
}

describe("styled shortcut tooltips", () => {
	let doc: DocumentStub;
	let node: ElementStub;
	let tip: ElementStub;
	let action: Exclude<ReturnType<typeof tooltip>, void>;
	beforeEach(() => {
		vi.useFakeTimers();
		doc = new DocumentStub();
		node = new ElementStub(doc);
		node.setAttribute("aria-describedby", "existing-help");
		action = tooltip(node as unknown as HTMLElement, "Accept (I)")!;
		tip = [...doc.children][0]!;
	});
	afterEach(() => { action.destroy?.(); vi.useRealTimers(); });
	it("preserves existing descriptions and gives shortcut text the styled tooltip class", () => {
		expect(tip.className).toBe("app-tooltip");
		expect(tip.getAttribute("role")).toBe("tooltip");
		expect(tip.textContent).toBe("Accept (I)");
		expect(node.getAttribute("aria-describedby")).toBe(`existing-help ${tip.id}`);
		expect(tip.hidden).toBe(true);
	});
	it("opens on keyboard focus and dismisses on Escape without handling the viewer's Escape", () => {
		node.dispatchEvent(new Event("focusin"));
		expect(tip.hidden).toBe(false);
		const escape = Object.assign(new Event("keydown"), { key: "Escape" });
		const stop = vi.spyOn(escape, "stopPropagation");
		doc.defaultView.dispatchEvent(escape);
		expect(tip.hidden).toBe(true);
		expect(stop).toHaveBeenCalledOnce();
	});
	it("delays hover, remains readable while hovered, and closes on leaving", () => {
		node.dispatchEvent(new Event("pointerenter"));
		vi.advanceTimersByTime(299);
		expect(tip.hidden).toBe(true);
		vi.advanceTimersByTime(1);
		expect(tip.hidden).toBe(false);
		node.dispatchEvent(new Event("pointerleave"));
		tip.dispatchEvent(new Event("pointerenter"));
		vi.advanceTimersByTime(100);
		expect(tip.hidden).toBe(false);
		tip.dispatchEvent(new Event("pointerleave"));
		vi.advanceTimersByTime(100);
		expect(tip.hidden).toBe(true);
	});
	it("updates dynamic text, removes empty descriptions, and cancels pending hover", () => {
		node.dispatchEvent(new Event("pointerenter"));
		action.update?.(undefined);
		vi.runAllTimers();
		expect(tip.hidden).toBe(true);
		expect(node.getAttribute("aria-describedby")).toBe("existing-help");
		action.update?.("Show suggestions (M)");
		node.dispatchEvent(new Event("focusin"));
		expect(tip.textContent).toBe("Show suggestions (M)");
		expect(tip.hidden).toBe(false);
	});
	it("only displays one tooltip, including nested controls", () => {
		node.dispatchEvent(new Event("focusin"));
		const other = new ElementStub(doc);
		const second = tooltip(other as unknown as HTMLElement, "Brush (B)")!;
		try {
			other.dispatchEvent(new Event("focusin"));
			expect(tip.hidden).toBe(true);
			expect([...doc.children].filter(node => !node.hidden)).toHaveLength(1);
		} finally { second.destroy?.(); }
	});
	it("removes its portal, listeners and pending timer on navigation", () => {
		node.dispatchEvent(new Event("pointerenter"));
		action.destroy?.();
		vi.runAllTimers();
		node.dispatchEvent(new Event("focusin"));
		expect(tip.hidden).toBe(true);
		expect(doc.children.size).toBe(0);
		expect(node.getAttribute("aria-describedby")).toBe("existing-help");
	});
});

describe("tooltip positioning", () => {
	it("stays on screen at either edge and flips above bottom controls", () => {
		expect(tooltipPosition({ left: 0, top: 20, bottom: 52, width: 32 }, { width: 150, height: 24 }, 400, 300)).toEqual({ left: 8, top: 60 });
		expect(tooltipPosition({ left: 370, top: 268, bottom: 300, width: 30 }, { width: 150, height: 24 }, 400, 300)).toEqual({ left: 242, top: 236 });
	});
});
