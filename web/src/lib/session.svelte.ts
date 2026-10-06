/** Who is signed in. Pages read `session.current`; null means signed out. */

import { api } from "./api";
import type { Session } from "./types";

class SessionState {
	current = $state<Session | null>(null);
	loaded = $state(false);

	async load(): Promise<Session | null> {
		try {
			this.current = await api<Session>("/api/auth/session");
		} catch {
			this.current = null;
		}
		this.loaded = true;
		return this.current;
	}

	async login(username: string, password: string, totp_code?: string): Promise<Session> {
		this.current = await api<Session>("/api/auth/login", {
			body: { username, password, totp_code: totp_code || undefined },
		});
		return this.current;
	}

	async signup(username: string, password: string, email?: string): Promise<Session> {
		this.current = await api<Session>("/api/auth/signup", {
			body: { username, password, email: email || undefined },
		});
		return this.current;
	}

	async logout(): Promise<void> {
		await api("/api/auth/logout", { method: "POST" });
		this.current = null;
	}
}

export const session = new SessionState();
