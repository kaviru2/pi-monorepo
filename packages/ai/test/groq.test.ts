import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { streamSimpleGroq } from "../src/providers/groq.js";
import type { Context, Model, SimpleStreamOptions } from "../src/types.js";

interface GroqPayload {
	model?: string;
	reasoning_format?: "parsed" | "raw" | "hidden";
	stream?: boolean;
}

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

async function capturePayload(model: Model<"groq-chat">, options?: SimpleStreamOptions): Promise<GroqPayload> {
	let capturedPayload: GroqPayload | undefined;

	const stream = streamSimpleGroq(model, makeContext(), {
		...options,
		apiKey: "fake-key",
		onPayload: (payload) => {
			capturedPayload = payload as GroqPayload;
			return payload;
		},
	});

	await stream.result().catch(() => {});

	if (!capturedPayload) {
		throw new Error("Expected payload to be captured");
	}

	return capturedPayload;
}

describe("Groq native provider reasoning format", () => {
	it("sets reasoning_format to parsed when reasoning is enabled", async () => {
		const baseModel = getModel("groq", "qwen/qwen3.6-27b");
		expect(baseModel).toBeDefined();
		if (!baseModel) return;

		const payload = await capturePayload(baseModel as Model<"groq-chat">, { reasoning: "medium" });
		expect(payload.reasoning_format).toBe("parsed");
	});

	it("sets reasoning_format to hidden when reasoning is off", async () => {
		const baseModel = getModel("groq", "qwen/qwen3.6-27b");
		expect(baseModel).toBeDefined();
		if (!baseModel) return;

		const payload = await capturePayload(baseModel as Model<"groq-chat">, { reasoning: "off" as any });
		expect(payload.reasoning_format).toBe("hidden");
	});
});
