import type { Action } from "svelte/action";

let nextId = 0;
const active = new WeakMap<Document, () => void>();

/** Native title tooltips cannot be styled; these keep shortcut letters unambiguous. */
export const tooltip: Action<HTMLElement, string | undefined> = (node, initial) => {
	const doc = node.ownerDocument;
	const win = doc.defaultView!;
	const tip = doc.createElement("div");
	tip.id = `app-tooltip-${++nextId}`;
	tip.className = "app-tooltip";
	tip.setAttribute("role", "tooltip");
	tip.hidden = true;
	doc.body.append(tip);
	let text = initial;
	let hovered = false;
	let focused = false;
	let timer: ReturnType<typeof setTimeout> | undefined;

	function cancel() { clearTimeout(timer); }
	function hide() {
		cancel(); tip.hidden = true;
		if (active.get(doc) === hide) active.delete(doc);
	}
	function describe(enabled: boolean) {
		const ids = (node.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(id => id && id !== tip.id);
		if (enabled) ids.push(tip.id);
		if (ids.length) node.setAttribute("aria-describedby", ids.join(" "));
		else node.removeAttribute("aria-describedby");
	}
	function update(value: string | undefined) {
		text = value;
		tip.textContent = value ?? "";
		describe(!!value);
		if (!value) hide();
		else if (!tip.hidden) position();
	}
	function position() {
		const anchor = node.getBoundingClientRect();
		const bounds = tip.getBoundingClientRect();
		const { left, top } = tooltipPosition(anchor, bounds, win.innerWidth, win.innerHeight);
		tip.style.left = `${left}px`;
		tip.style.top = `${top}px`;
	}
	function show() {
		cancel();
		if (!text) return;
		active.get(doc)?.();
		active.set(doc, hide);
		tip.hidden = false;
		position();
	}
	function enter() { hovered = true; cancel(); timer = setTimeout(show, 300); }
	function leave() { hovered = false; cancel(); if (!focused) timer = setTimeout(hide, 100); }
	function focus() { focused = true; show(); }
	function blur() { focused = false; if (!hovered) hide(); }
	function enterTip() { hovered = true; cancel(); }
	function key(event: KeyboardEvent) {
		if (event.key !== "Escape" || tip.hidden) return;
		hide();
		// Dismissing a tooltip must not also switch the current drawing tool.
		event.stopPropagation();
	}
	const listeners = [
		[node, "pointerenter", enter], [node, "pointerleave", leave],
		[node, "focusin", focus], [node, "focusout", blur], [node, "pointerdown", hide],
		[tip, "pointerenter", enterTip], [tip, "pointerleave", leave],
		[win, "keydown", key], [win, "resize", hide], [doc, "scroll", hide],
	] as const;
	for (const [target, event, callback] of listeners) target.addEventListener(event, callback as EventListener, { capture: true });
	update(initial);
	return {
		update,
		destroy() {
			hide();
			describe(false);
			for (const [target, event, callback] of listeners) target.removeEventListener(event, callback as EventListener, { capture: true });
			tip.remove();
		},
	};
};

/** Keep tooltips in the viewport, above the control when there is no room below. */
export function tooltipPosition(
	anchor: Pick<DOMRect, "left" | "top" | "bottom" | "width">,
	tip: Pick<DOMRect, "width" | "height">,
	width: number,
	height: number,
) {
	const margin = 8;
	return {
		left: Math.max(margin, Math.min(anchor.left + (anchor.width - tip.width) / 2, width - tip.width - margin)),
		top: Math.max(margin, anchor.bottom + margin + tip.height <= height - margin ? anchor.bottom + margin : anchor.top - tip.height - margin),
	};
}
