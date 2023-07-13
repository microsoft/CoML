import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage, BaseMessage, AIMessage } from "langchain/schema";

import { openAIApiKey } from "./apiKey";

export async function chatWithGPT(messages: BaseMessage[]): Promise<AIMessage> {
    const model = new ChatOpenAI({
        openAIApiKey: openAIApiKey,
        temperature: 0.9,
        topP: 1,
        modelName: "gpt-3.5-turbo"
    });
    return await model.call([
        new SystemMessage("You are a helpful assistant."),
        ...messages
    ]);
}
