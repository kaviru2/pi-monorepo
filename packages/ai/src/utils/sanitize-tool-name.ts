/**
 * Utility to sanitize tool call names emitted by LLMs.
 * Smaller/open-weights models (e.g. Qwen, Llama, DeepSeek) fine-tuned on ChatML
 * or multi-channel outputs sometimes append ChatML special tokens or channel suffixes
 * like `<|channel|>commentary`, `<|thought|>`, `<|call|>`, `<|im_end|>`, etc.
 */

export function sanitizeToolName(name: string): string {
	if (!name) return "";

	let cleaned = name;

	// Strip ChatML special tokens and any characters following them (e.g. <|channel|>commentary -> "")
	cleaned = cleaned.replace(/<\|.*?\|>.*$/, "");

	// Strip XML-style tags or brackets if present
	cleaned = cleaned.replace(/<[^>]*>/g, "");

	// Trim extraneous whitespace or punctuation suffixes
	cleaned = cleaned.trim();

	return cleaned;
}
