import { Groq } from "groq-sdk";
import { getEnvApiKey } from "./src/env-api-keys.js";

async function main() {
    const client = new Groq();
    const stream = await client.chat.completions.create({
        model: "llama3-8b-8192",
        messages: [{ role: "user", content: "Hello, world!" }],
        stream: true,
        stream_options: { include_usage: true }
    });

    for await (const chunk of stream) {
        if ((chunk as any).usage || (chunk as any).x_groq?.usage) {
            console.log("Usage chunk:", JSON.stringify(chunk, null, 2));
        }
    }
}
main().catch(console.error);
