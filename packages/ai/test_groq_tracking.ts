import { streamSimpleGroq } from "./src/providers/groq.js";
import { getModel } from "./src/models.js";
import type { Model, Context } from "./src/types.js";

async function main() {
    const model = getModel("groq", "llama-3.3-70b-versatile") || getModel("groq", "llama3-8b-8192") || getModel("groq", "llama-3.1-8b-instant");
    if (!model) throw new Error("Model not found");

    const context: Context = {
        messages: [{ role: "user", content: "Tell me a short joke about a programmer.", timestamp: Date.now() }]
    };

    const stream = streamSimpleGroq(model as Model<"groq-chat">, context, {
        apiKey: process.env.GROQ_API_KEY
    });

    console.log(`Streaming response from model: ${model.id}...`);

    const result = await stream.result();
    console.log("\n\nFull Result Object:", JSON.stringify(result, null, 2));
}

main().catch(console.error);
