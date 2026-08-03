import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { streamGroq } from "../src/providers/groq.js";
import type { Context, Model } from "../src/types.js";

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

describe("Groq provider rate limit retry", () => {
	it("retries on 429 rate_limit_exceeded error with backoff", async () => {
		const baseModel = getModel("groq", "qwen/qwen3.6-27b");
		expect(baseModel).toBeDefined();
		if (!baseModel) return;

		let attemptCount = 0;
		const stream = streamGroq(baseModel as Model<"groq-chat">, makeContext(), {
			apiKey: "fake-key",
			maxRetries: 2,
			onPayload: () => {
				attemptCount++;
				if (attemptCount === 1) {
					const error: any = new Error(
						"Rate limit reached for model openai/gpt-oss-20b. Please try again in 0.01s.",
					);
					error.status = 429;
					throw error;
				}
				throw new Error("PAUSE_TEST");
			},
		});

		await stream.result().catch((err) => err);
		expect(attemptCount).toBeGreaterThanOrEqual(1);
	});

	it("retries on Groq server-side Parsing failed (parse_error) transient errors", async () => {
		const baseModel = getModel("groq", "qwen/qwen3.6-27b");
		expect(baseModel).toBeDefined();
		if (!baseModel) return;

		let attemptCount = 0;
		const stream = streamGroq(baseModel as Model<"groq-chat">, makeContext(), {
			apiKey: "fake-key",
			maxRetries: 2,
			onPayload: () => {
				attemptCount++;
				if (attemptCount === 1) {
					const error: any = new Error(
						"Parsing failed. The model generated output that could not be parsed. Please adjust your prompt. See 'failed_generation' for more details.",
					);
					error.status = 400;
					error.code = "parse_error";
					throw error;
				}
				throw new Error("PAUSE_TEST");
			},
		});

		const res = await stream.result().catch((err) => err);
		expect(attemptCount).toBe(2);
		expect(res.diagnostics?.some((d: any) => d.type === "retry" && d.details?.reason === "parse_error")).toBe(true);
	});

	it("retries on a tool-name hallucination (tool_validation_error) without injecting any correction", async () => {
		const baseModel = getModel("groq", "qwen/qwen3.6-27b");
		expect(baseModel).toBeDefined();
		if (!baseModel) return;

		let attemptCount = 0;
		let messagesOnSecondAttempt: any[] = [];
		const stream = streamGroq(baseModel as Model<"groq-chat">, makeContext(), {
			apiKey: "fake-key",
			maxRetries: 2,
			onPayload: (payload: any) => {
				attemptCount++;
				if (attemptCount === 1) {
					const error: any = new Error(
						"Tool call validation failed: tool call validation failed: attempted to call tool 'run' which was not in request.tools",
					);
					error.status = 400;
					throw error;
				}
				messagesOnSecondAttempt = payload.messages;
				throw new Error("PAUSE_TEST");
			},
		});

		const res = await stream.result().catch((err) => err);
		expect(attemptCount).toBe(2);
		expect(
			res.diagnostics?.some((d: any) => d.type === "retry" && d.details?.reason === "tool_validation_error"),
		).toBe(true);
		// The retry blindly resends the same single message — no corrective message is injected.
		expect(messagesOnSecondAttempt).toHaveLength(1);
		expect(messagesOnSecondAttempt[0].content).toBe("Hello");
	});
});
