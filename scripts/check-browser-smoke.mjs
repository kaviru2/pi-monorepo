import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { build } from "esbuild";

const outputPath = join(tmpdir(), "pi-browser-smoke.js");
const errorLogPath = join(tmpdir(), "pi-browser-smoke-errors.log");

try {
	await build({
		entryPoints: ["scripts/browser-smoke-entry.ts"],
		bundle: true,
		platform: "browser",
		format: "esm",
		logLevel: "silent",
		outfile: outputPath,
		// @mistralai/mistralai statically imports these for optional OTEL tracing (see
		// peerDependenciesMeta in its package.json — all marked optional, none installed here since
		// nothing in this repo actually uses telemetry). A static bundler resolves the import
		// regardless of whether the code path runs, so they must be marked external rather than left
		// for npm to install just to satisfy this smoke build.
		external: [
			"@opentelemetry/api",
			"@opentelemetry/exporter-trace-otlp-http",
			"@opentelemetry/resources",
			"@opentelemetry/sdk-trace-base",
		],
	});
	process.exit(0);
} catch (error) {
	let detailedErrors = "";
	if (error && typeof error === "object" && "errors" in error && Array.isArray(error.errors)) {
		detailedErrors = error.errors
			.map((entry) => {
				const location = entry.location
					? `${entry.location.file}:${entry.location.line}:${entry.location.column}`
					: "";
				return [location, entry.text].filter(Boolean).join(" ");
			})
			.join("\n");
	}

	const baseError = error instanceof Error ? (error.stack ?? error.message) : String(error);
	writeFileSync(errorLogPath, [detailedErrors, baseError].filter(Boolean).join("\n\n"), "utf-8");
	console.error(`Browser smoke check failed. See ${errorLogPath}`);
	process.exit(1);
}
