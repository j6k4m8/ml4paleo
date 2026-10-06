/** Where the user is, for the top bar's breadcrumbs; each page sets its own. */

export interface Crumb {
	label: string;
	href?: string;
}

class Crumbs {
	items = $state<Crumb[]>([]);

	set(items: Crumb[]): void {
		this.items = items;
	}
}

export const crumbs = new Crumbs();
