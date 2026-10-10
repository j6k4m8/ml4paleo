import { BACKGROUND_CLASS } from "./background";
import { classOpacity, readClassStyles, type ClassStyle, type ClassStyles } from "./class-display";

export class ClassDisplay {
	styles: ClassStyles = $state.raw({});
	readonly key: string;

	constructor(project: string, user: string) {
		this.key = `m4p.classes:${encodeURIComponent(user)}:${encodeURIComponent(project)}`;
		try {
			this.styles = readClassStyles(JSON.parse(localStorage.getItem(this.key) ?? "{}"));
		} catch { /* Unavailable storage or old/broken preferences: use defaults. */ }
	}

	get backgroundColor(): string {
		return this.styles[1]?.color ?? BACKGROUND_CLASS.color;
	}

	set(value: number, changes: ClassStyle): void {
		this.styles = readClassStyles({ ...this.styles, [value]: { ...this.styles[value], ...changes } });
		try { localStorage.setItem(this.key, JSON.stringify(this.styles)); } catch { /* Still works in memory. */ }
	}

	/** A fresh painted stroke should never silently disappear into a hidden class. */
	reveal(value: number): boolean {
		if (classOpacity(value, this.styles) > 0) return false;
		this.set(value, { visible: true, ...(this.styles[value]?.opacity === 0 ? { opacity: 1 } : {}) });
		return true;
	}
}
