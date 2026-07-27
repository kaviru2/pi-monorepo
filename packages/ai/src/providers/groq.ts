import Groq from "groq-sdk";
import type {
	ChatCompletionAssistantMessageParam,
	ChatCompletionChunk,
	ChatCompletionContentPart,
	ChatCompletionMessageParam,
} from "groq-sdk/resources/chat/completions.js";
import { getEnvApiKey } from "../env-api-keys.js";
import { calculateCost, clampThinkingLevel } from "../models.js";
import type {
	AssistantMessage,
	Context,
	Message,
	Model,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "../types.js";
import { appendAssistantMessageDiagnostic, createAssistantMessageDiagnostic } from "../utils/diagnostics.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { headersToRecord } from "../utils/headers.js";
import { parseStreamingJson } from "../utils/json-parse.js";
import { sanitizeToolName } from "../utils/sanitize-tool-name.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { buildBaseOptions } from "./simple-options.js";
import { transformMessages } from "./transform-messages.js";

export interface GroqOptions extends StreamOptions {
	reasoningFormat?: "parsed" | "raw" | "hidden";
	temperature?: number;
	topP?: number;
}

/**
 * Stream responses from Groq using native groq-sdk.
 */
export const streamGroq: StreamFunction<"groq-chat", GroqOptions> = (
	model: Model<"groq-chat">,
	context: Context,
	options?: GroqOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey || getEnvApiKey(model.provider) || "";

			let baseURL = model.baseUrl || "https://api.groq.com";
			if (baseURL.endsWith("/openai/v1")) {
				baseURL = baseURL.slice(0, -"/openai/v1".length);
			}

			const client = new Groq({
				apiKey,
				baseURL,
				defaultHeaders: options?.headers,
			});

			const supportsImages = Boolean(model.input?.includes("image"));
			const messages = toChatMessages(transformMessages(context.messages, model), supportsImages);
			const systemPrompt = context.systemPrompt;

			const formattedMessages: ChatCompletionMessageParam[] = [];
			if (systemPrompt) {
				formattedMessages.push({ role: "system", content: systemPrompt });
			}
			formattedMessages.push(...messages);

			const params: Record<string, unknown> = {
				model: model.id,
				messages: formattedMessages,
				stream: true,
				stream_options: { include_usage: true },
			};

			if (options?.maxTokens) {
				params.max_completion_tokens = options.maxTokens;
			}
			if (options?.temperature !== undefined) {
				params.temperature = options.temperature;
			}
			if (options?.topP !== undefined) {
				params.top_p = options.topP;
			}
			if (context.tools && context.tools.length > 0) {
				params.tools = context.tools.map((t) => ({
					type: "function",
					function: {
						name: t.name,
						description: t.description,
						parameters: t.parameters,
					},
				}));
			}

			if (model.reasoning) {
				if (options?.reasoningFormat) {
					params.reasoning_format = options.reasoningFormat;
				} else {
					params.reasoning_format = "parsed";
				}
			}

			const requestOptions = {
				...(options?.signal ? { signal: options.signal } : {}),
				...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
				maxRetries: 0, // Handle retries manually below to control backoff & logging
			};

			const maxRetries = options?.maxRetries ?? 5;
			let attempts = 0;
			let startEmitted = false;

			// Retry loop covers both the HTTP request AND the full streaming consumption.
			// This ensures mid-stream Groq errors (tool name validation, parse failures) are
			// retried just like pre-stream errors.
			while (true) {
				// Reset mutable output state before each attempt so replayed events are clean.
				output.content = [];
				output.stopReason = "stop";
				output.errorMessage = undefined;
				output.responseModel = undefined;
				output.responseId = undefined;
				output.usage = {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				};

				let currentTextBlock: TextContent | undefined;
				let currentThinkingBlock: ThinkingContent | undefined;
				const toolCallsByCallId = new Map<
					string,
					{
						toolCall: ToolCall;
						index: number;
						nameBuffer: string;
						argsBuffer: string;
						headerEmitted: boolean;
					}
				>();
				const toolCallsByIndex = new Map<number, string>();

				try {
					let payload = params;
					if (options?.onPayload) {
						const next = await options.onPayload(payload, model);
						if (next !== undefined) payload = next as typeof params;
					}

					const res = await client.chat.completions.create(payload as any, requestOptions).withResponse();
					await options?.onResponse?.(
						{ status: res.response.status, headers: headersToRecord(res.response.headers) },
						model,
					);

					if (!startEmitted) {
						stream.push({ type: "start", partial: output });
						startEmitted = true;
					}

					for await (const chunk of res.data as unknown as AsyncIterable<ChatCompletionChunk>) {
						if (options?.signal?.aborted) {
							throw new Error("Request was aborted");
						}

						if (chunk.model && !output.responseModel) {
							output.responseModel = chunk.model;
						}
						if (chunk.id && !output.responseId) {
							output.responseId = chunk.id;
						}

						const choice = chunk.choices?.[0];
						if (choice?.delta) {
							const delta = choice.delta as any;

							// Native Groq reasoning delta parsing
							const reasoningText = delta.reasoning || delta.reasoning_content;
							if (reasoningText) {
								if (!currentThinkingBlock) {
									currentThinkingBlock = { type: "thinking", thinking: "" };
									output.content.push(currentThinkingBlock);
									const contentIndex = output.content.length - 1;
									stream.push({ type: "thinking_start", contentIndex, partial: output });
								}
								const cleanDelta = sanitizeSurrogates(reasoningText);
								currentThinkingBlock.thinking += cleanDelta;
								const contentIndex = output.content.indexOf(currentThinkingBlock);
								stream.push({ type: "thinking_delta", contentIndex, delta: cleanDelta, partial: output });
							}

							// Standard Content Text
							if (delta.content) {
								if (currentThinkingBlock) {
									const contentIndex = output.content.indexOf(currentThinkingBlock);
									stream.push({
										type: "thinking_end",
										contentIndex,
										content: currentThinkingBlock.thinking,
										partial: output,
									});
									currentThinkingBlock = undefined;
								}

								if (!currentTextBlock) {
									currentTextBlock = { type: "text", text: "" };
									output.content.push(currentTextBlock);
									const contentIndex = output.content.length - 1;
									stream.push({ type: "text_start", contentIndex, partial: output });
								}
								const cleanDelta = sanitizeSurrogates(delta.content);
								currentTextBlock.text += cleanDelta;
								const contentIndex = output.content.indexOf(currentTextBlock);
								stream.push({ type: "text_delta", contentIndex, delta: cleanDelta, partial: output });
							}

							// Tool Call Deltas
							if (delta.tool_calls) {
								if (currentThinkingBlock) {
									const contentIndex = output.content.indexOf(currentThinkingBlock);
									stream.push({
										type: "thinking_end",
										contentIndex,
										content: currentThinkingBlock.thinking,
										partial: output,
									});
									currentThinkingBlock = undefined;
								}
								if (currentTextBlock) {
									const contentIndex = output.content.indexOf(currentTextBlock);
									stream.push({
										type: "text_end",
										contentIndex,
										content: currentTextBlock.text,
										partial: output,
									});
									currentTextBlock = undefined;
								}

								for (const tcDelta of delta.tool_calls) {
									const index = tcDelta.index;
									let callId = tcDelta.id;

									if (callId) {
										toolCallsByIndex.set(index, callId);
									} else {
										callId = toolCallsByIndex.get(index);
									}

									if (!callId) continue;

									let state = toolCallsByCallId.get(callId);
									if (!state) {
										const initialRawName = tcDelta.function?.name || "";
										const toolCall: ToolCall = {
											type: "toolCall",
											id: callId,
											name: sanitizeToolName(initialRawName),
											arguments: {},
										};
										output.content.push(toolCall);
										const contentIndex = output.content.length - 1;
										state = {
											toolCall,
											index: contentIndex,
											nameBuffer: initialRawName,
											argsBuffer: "",
											headerEmitted: false,
										};
										toolCallsByCallId.set(callId, state);
									} else if (tcDelta.function?.name) {
										state.nameBuffer += tcDelta.function.name;
										state.toolCall.name = sanitizeToolName(state.nameBuffer);
									}

									if (state.toolCall.name && !state.headerEmitted) {
										stream.push({ type: "toolcall_start", contentIndex: state.index, partial: output });
										state.headerEmitted = true;
									}

									if (tcDelta.function?.arguments) {
										const argChunk = sanitizeSurrogates(tcDelta.function.arguments);
										state.argsBuffer += argChunk;
										if (state.headerEmitted) {
											stream.push({
												type: "toolcall_delta",
												contentIndex: state.index,
												delta: argChunk,
												partial: output,
											});
										}
									}
								}
							}
						}

						if (choice?.finish_reason) {
							switch (choice.finish_reason) {
								case "stop":
									output.stopReason = "stop";
									break;
								case "length":
									output.stopReason = "length";
									break;
								case "tool_calls":
									output.stopReason = "toolUse";
									break;
								default:
									output.stopReason = "stop";
							}
						}

						const rawChunk = chunk as any;
						if (rawChunk.usage) {
							output.usage.input = rawChunk.usage.prompt_tokens || 0;
							output.usage.output = rawChunk.usage.completion_tokens || 0;
							output.usage.totalTokens = rawChunk.usage.total_tokens || 0;
							calculateCost(model, output.usage);
						}
					}

					// Finalize active blocks
					if (currentThinkingBlock) {
						const contentIndex = output.content.indexOf(currentThinkingBlock);
						stream.push({
							type: "thinking_end",
							contentIndex,
							content: currentThinkingBlock.thinking,
							partial: output,
						});
					}
					if (currentTextBlock) {
						const contentIndex = output.content.indexOf(currentTextBlock);
						stream.push({ type: "text_end", contentIndex, content: currentTextBlock.text, partial: output });
					}

					for (const state of toolCallsByCallId.values()) {
						state.toolCall.arguments = parseStreamingJson(state.argsBuffer) || {};
						stream.push({
							type: "toolcall_end",
							contentIndex: state.index,
							toolCall: state.toolCall,
							partial: output,
						});
					}

					const sr = output.stopReason as string;
					const doneReason = (sr === "aborted" || sr === "error" ? "stop" : output.stopReason) as
						| "stop"
						| "length"
						| "toolUse";
					stream.push({ type: "done", reason: doneReason, message: output });
					stream.end();
					return; // success — exit retry loop
				} catch (err) {
					if (options?.signal?.aborted) {
						output.stopReason = "aborted";
						output.errorMessage = err instanceof Error ? err.message : String(err);
						stream.push({ type: "error", reason: "aborted", error: output });
						stream.end();
						return;
					}
					const retryInfo = isRateLimitOrTransientError(err);
					if (attempts < maxRetries && retryInfo.isRetryable) {
						attempts++;
						const parsedMs = parseRetryDelayMs(err);
						const baseWaitMs = retryInfo.reason === "rate_limit" ? 10000 : 2000;
						const backoffMs = Math.max(parsedMs || 0, Math.round(baseWaitMs * 1.5 ** (attempts - 1)));

						const errorText = err instanceof Error ? err.message : String(err);
						console.warn(
							`[Groq Retry] Attempt ${attempts}/${maxRetries} failed (${retryInfo.reason}). Retrying in ${(backoffMs / 1000).toFixed(2)}s... Error: ${errorText}`,
						);

						appendAssistantMessageDiagnostic(
							output,
							createAssistantMessageDiagnostic("retry", err, {
								attempt: attempts,
								maxRetries,
								backoffMs,
								reason: retryInfo.reason,
							}),
						);

						await sleep(backoffMs, options?.signal);
						continue;
					}
					// Non-retryable or exhausted retries
					output.stopReason = "error";
					output.errorMessage = err instanceof Error ? err.message : JSON.stringify(err);
					stream.push({ type: "error", reason: "error", error: output });
					stream.end();
					return;
				}
			} // end while
		} catch (error) {
			// Covers unexpected throws from setup code before the while loop (e.g. transformMessages)
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

export const streamSimpleGroq: StreamFunction<"groq-chat", SimpleStreamOptions> = (
	model: Model<"groq-chat">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	const apiKey = options?.apiKey || getEnvApiKey(model.provider);
	if (!apiKey) {
		throw new Error(`No API key for provider: ${model.provider}`);
	}

	const base = buildBaseOptions(model, options, apiKey);
	const clampedReasoning = options?.reasoning ? clampThinkingLevel(model, options.reasoning) : undefined;
	const reasoningFormat = clampedReasoning === "off" ? "hidden" : "parsed";

	return streamGroq(model, context, {
		...base,
		reasoningFormat,
	});
};

function toChatMessages(messages: Message[], supportsImages: boolean): ChatCompletionMessageParam[] {
	const result: ChatCompletionMessageParam[] = [];

	for (const msg of messages) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				result.push({ role: "user", content: msg.content });
			} else {
				const parts: ChatCompletionContentPart[] = [];
				for (const part of msg.content) {
					if (part.type === "text") {
						parts.push({ type: "text", text: part.text });
					} else if (part.type === "image" && supportsImages) {
						parts.push({
							type: "image_url",
							image_url: { url: `data:${part.mimeType};base64,${part.data}` },
						});
					}
				}
				result.push({ role: "user", content: parts });
			}
		} else if (msg.role === "assistant") {
			let text = "";
			const toolCalls: ChatCompletionAssistantMessageParam["tool_calls"] = [];

			for (const block of msg.content) {
				if (block.type === "text") {
					text += block.text;
				} else if (block.type === "toolCall") {
					toolCalls.push({
						id: block.id,
						type: "function",
						function: {
							name: block.name,
							arguments: JSON.stringify(block.arguments),
						},
					});
				}
			}

			const param: ChatCompletionAssistantMessageParam = { role: "assistant" };
			if (text) param.content = text;
			if (toolCalls.length > 0) param.tool_calls = toolCalls;
			result.push(param);
		} else if (msg.role === "toolResult") {
			const textParts: string[] = [];
			for (const block of msg.content) {
				if (block.type === "text") {
					textParts.push(block.text);
				}
			}
			result.push({
				role: "tool",
				tool_call_id: msg.toolCallId,
				content: textParts.join("\n"),
			});
		}
	}

	return result;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new Error("Request was aborted"));
			return;
		}
		const timeout = setTimeout(resolve, ms);
		signal?.addEventListener("abort", () => {
			clearTimeout(timeout);
			reject(new Error("Request was aborted"));
		});
	});
}

function isRateLimitOrTransientError(error: unknown): { isRetryable: boolean; reason: string } {
	if (!error || typeof error !== "object") return { isRetryable: false, reason: "none" };
	const status = (error as any).status || (error as any).statusCode;
	const code = (error as any).code || (error as any).error?.code;
	const msg = String((error as any).message || (error as any).error?.message || error);

	if (status === 429 || /rate_limit_exceeded|rate.?limit|too many requests|please try again in/i.test(msg)) {
		return { isRetryable: true, reason: "rate_limit" };
	}

	if (
		code === "parse_error" ||
		code === "failed_generation" ||
		/parsing failed|failed_generation|could not be parsed/i.test(msg)
	) {
		return { isRetryable: true, reason: "parse_error" };
	}

	if (
		status === 500 ||
		status === 502 ||
		status === 503 ||
		status === 504 ||
		/server error|overloaded|service.?unavailable/i.test(msg)
	) {
		return { isRetryable: true, reason: "server_error" };
	}

	return { isRetryable: false, reason: "none" };
}

function parseRetryDelayMs(error: unknown): number | undefined {
	const msg = String((error as any)?.message || error);
	const match = msg.match(/try again in ([\d.]+)s/i);
	if (match?.[1]) {
		const sec = Number.parseFloat(match[1]);
		if (!Number.isNaN(sec)) {
			return Math.ceil(sec * 1000);
		}
	}
	return undefined;
}
