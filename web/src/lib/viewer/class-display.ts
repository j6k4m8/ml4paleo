/** Browser-local display preferences, never labels or training inputs. */
export interface ClassStyle {
	visible?: boolean;
	opacity?: number;
	/** Only Background uses a local color; foreground colors belong to the project. */
	color?: string;
}

export type ClassStyles = Record<number, ClassStyle>;

export function classOpacity(value: number, styles: ClassStyles): number {
	const style = styles[value];
	if (style?.visible === false) return 0;
	const opacity = style?.opacity;
	return typeof opacity === "number" && Number.isFinite(opacity)
		? Math.round(Math.max(0, Math.min(1, opacity)) * 100) / 100
		: 1;
}

/** Exclude invisible classes from decisions without altering the source data. */
export function displayedValues(values: Uint8Array, styles: ClassStyles): Uint8Array {
	return values.map((value) => value > 0 && value < 255 && classOpacity(value, styles) > 0 ? value : 0);
}

export function readClassStyles(value: unknown): ClassStyles {
	if (!value || typeof value !== "object" || Array.isArray(value)) return {};
	const styles: ClassStyles = {};
	for (const [key, candidate] of Object.entries(value)) {
		const id = Number(key);
		if (!Number.isInteger(id) || id < 1 || id > 254 || !candidate || typeof candidate !== "object") continue;
		const raw = candidate as ClassStyle;
		const style: ClassStyle = {};
		if (typeof raw.visible === "boolean") style.visible = raw.visible;
		if (typeof raw.opacity === "number" && Number.isFinite(raw.opacity)) {
			style.opacity = classOpacity(id, { [id]: { opacity: raw.opacity } });
		}
		if (id === 1 && typeof raw.color === "string" && /^#[0-9a-f]{6}$/i.test(raw.color)) style.color = raw.color.toLowerCase();
		styles[id] = style;
	}
	return styles;
}
