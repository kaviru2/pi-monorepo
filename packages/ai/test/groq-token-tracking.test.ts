// We need to mock groq-sdk
import { describe, expect, it, vi } from "vitest";
import { getModel } from "../src/models.js";
import { streamSimpleGroq } from "../src/providers/groq.js";
import type { Model } from "../src/types.js";

// This is a minimal mock for the groq-sdk
vi.mock("groq-sdk", () => {
	const Groq = vi.fn();
	Groq.prototype.chat = {
		completions: {
			create: vi.fn().mockImplementation(() => {
				return {
					withResponse: async () => {
						return {
							response: { status: 200, headers: new Headers() },
							data: (async function* () {
								// Mocked streaming chunks with x_groq.usage
								yield {
									id: "mock-1",
									model: "llama-3.3-70b-versatile",
									choices: [{ delta: { content: "Here is a " }, finish_reason: null }],
								};
								yield {
									id: "mock-1",
									model: "llama-3.3-70b-versatile",
									choices: [{ delta: { content: "joke!" }, finish_reason: "stop" }],
									x_groq: {
										usage: {
											prompt_tokens: 15,
											completion_tokens: 5,
											total_tokens: 20,
										},
									},
								};
							})(),
						};
					},
				};
			}),
		},
	};
	return { default: Groq, Groq };
});

describe("Groq token tracking", () => {
	it("should capture usage from x_groq.usage during streams", async () => {
		const model = getModel("groq", "llama-3.3-70b-versatile");
		expect(model).toBeDefined();
		if (!model) return;

		const stream = streamSimpleGroq(model as Model<"groq-chat">, { messages: [] }, { apiKey: "fake" });
		const result = await stream.result();

		expect(result.content[0].type).toBe("text");
		expect((result.content[0] as any).text).toBe("Here is a joke!");

		// Check if the usage was properly tracked
		expect(result.usage.input).toBe(15);
		expect(result.usage.output).toBe(5);
		expect(result.usage.totalTokens).toBe(20);
	});
});
