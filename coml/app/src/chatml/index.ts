import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage, BaseMessage, AIMessage } from "langchain/schema";

import { loadDatabase } from "./database";
import { openAIApiKey } from "./apiKey";
import { Module, Solution, VerifiedAlgorithm, Dataset, Model, TaskType } from "./types";

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

export async function suggestMachineLearningModule(
    existingModules: Module[],
    targetRole: string,
    targetSchemaId: string
): Promise<Module> {
    
}

interface Example {
    input: Module[];
    output: {
        candidate: Module;
        feedback?: string;
    };
}

async function findExamples(
    existingModules: Module[],
    targetRole: string,
    targetSchemaId: string | undefined
): Promise<Example[]> {
    const database = await loadDatabase();

    // Filter and compute matching scores.
    const solutionsAsIO: { input: Module[], output: Module, matchingScore: number }[] = [];
    for (const solution of database.solutions) {
        const exampleOutputModule = solution.modules.find((module) => {
            return module.role === targetRole && (
                targetSchemaId === undefined || (
                    module.role === "verifiedAlgorithm" &&
                    (module.module as any).schema === targetSchemaId
                )
            )
        });
        if (exampleOutputModule === undefined) {
            continue;
        }
        const exampleInputModules = solution.modules.filter((module) => module !== exampleOutputModule);
        let matchingScore = 0.;
        for (const exampleInputModule of exampleInputModules) {
            for (const existingModule of existingModules) {
                matchingScore += moduleSimilarity(exampleInputModule, existingModule);
            }
        }
        solutionsAsIO.push({
            input: exampleInputModules,
            output: exampleOutputModule,
            matchingScore: matchingScore
        });
    }

    
}

function moduleSimilarity(module1: Module, module2: Module): number {
    if (module1.role === module2.role) {
        if (module1.role === "dataset") {
            return textSimilarity(
                (module1.module as Dataset).description,
                (module2.module as Dataset).description
            );
        } else if (module1.role === "model") {
            return textSimilarity(
                (module1.module as Model).description,
                (module2.module as Model).description
            );
        } else if (module1.role === "taskType") {
            return textSimilarity(
                (module1.module as TaskType).description,
                (module2.module as TaskType).description
            );
        }
        // Ignore other types of modules.
    }
    return 0.;
}

function textSimilarity(text1: string, text2: string): number {
    return 0.;
}