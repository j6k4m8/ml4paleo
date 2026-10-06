import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
	plugins: [
		sveltekit({
			// A static single-page app: the API server serves index.html for
			// every path that isn't a file, and the app routes in the browser.
			adapter: adapter({ fallback: "index.html" }),
		}),
	],
	server: {
		// `npm run dev` talks to an API server running on this machine.
		proxy: { "/api": "http://localhost:8000" },
	},
	test: {
		include: ["src/**/*.test.ts"],
	},
});
