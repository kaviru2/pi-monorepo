import { describe, expect, it } from "vitest";
import { sanitizeToolName } from "../src/utils/sanitize-tool-name.js";

describe("sanitizeToolName", () => {
	it("strips ChatML channel tags and trailing commentary", () => {
		expect(sanitizeToolName("interactive_shell<|channel|>commentary")).toBe("interactive_shell");
		expect(sanitizeToolName("web_search<|channel|>call")).toBe("web_search");
		expect(sanitizeToolName("read_file<|thought|>extra_tokens")).toBe("read_file");
		expect(sanitizeToolName("run_command<|im_end|>")).toBe("run_command");
	});

	it("strips XML tags and brackets", () => {
		expect(sanitizeToolName("interactive_shell<commentary>")).toBe("interactive_shell");
		expect(sanitizeToolName("<tool>read_file</tool>")).toBe("read_file");
	});

	it("trims whitespace", () => {
		expect(sanitizeToolName("  interactive_shell  ")).toBe("interactive_shell");
	});

	it("handles empty or standard tool names untouched", () => {
		expect(sanitizeToolName("")).toBe("");
		expect(sanitizeToolName("interactive_shell")).toBe("interactive_shell");
		expect(sanitizeToolName("math_operation")).toBe("math_operation");
	});

	it("handles tool names accumulated across streaming deltas", () => {
		let buffer = "";
		const deltas = ["inter", "active_", "shell", "<|channel|>", "commentary"];
		for (const d of deltas) {
			buffer += d;
		}
		expect(sanitizeToolName(buffer)).toBe("interactive_shell");
	});
});
