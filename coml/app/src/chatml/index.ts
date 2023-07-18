import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage, BaseMessage, AIMessage } from "langchain/schema";

import { loadDatabase } from "./database";
import { openAIApiKey } from "./apiKey";
import { queryEmbedding } from "./embedding";
import { Module, Solution, VerifiedAlgorithm, Dataset, Model, TaskType, Metric, Knowledge } from "./types";

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
) {
  const examples = findExamples(existingModules, targetRole, targetSchemaId);
  const knowledges = findKnowledge(existingModules, targetRole, targetSchemaId);
  console.log((await examples).slice(0, 10));
  console.log((await knowledges).slice(0, 10));
}

interface Example {
  input: Module[];
  output: {
    candidate: Module;
    metric: number;
    feedback?: string;
  }[];
  matchingScore: number;
}

async function findExamples(
  existingModules: Module[],
  targetRole: string,
  targetSchemaId: string | undefined
): Promise<Example[]> {
  const database = await loadDatabase();

  // Step 1: Filter and compute matching scores.
  const solutionsAsIO = new Map<string, Example>();
  for (const solution of database.solutions) {
    // 1.a: Find an example output module.
    const exampleOutputModule = findTargetModule(solution.modules, targetRole, targetSchemaId);
    if (exampleOutputModule === undefined) {
      continue;
    }

    // 1.b: Must have at least one overlapping input role.
    const exampleInputModules = solution.modules.filter((module) => module !== exampleOutputModule);
    if (!hasOverlapModule(exampleInputModules, existingModules)) {
      continue;
    }

    // 1.c: Compute matching score and adding to mapping.
    const inputModulesAsJson = JSON.stringify(exampleInputModules);
    if (!solutionsAsIO.has(inputModulesAsJson)) {
      solutionsAsIO.set(inputModulesAsJson, {
        input: exampleInputModules,
        output: [],
        matchingScore: await moduleSimilarity(exampleInputModules, existingModules),
      });
    }
    solutionsAsIO.get(inputModulesAsJson)!.output.push({
      candidate: exampleOutputModule,
      metric: typeof solution.metrics === "number" ? solution.metrics : 0.,
      feedback: typeof solution.metrics !== "number" ? JSON.stringify(solution.metrics) : undefined
    });
  }

  // Step 2: Get the group of examples, sorted by matching score.
  const unsortedExamples = Array.from(solutionsAsIO.values()).map((item) => {
    item.output = item.output.sort((a, b) => (b.metric - a.metric));
    return item;
  });
  return unsortedExamples.sort((a, b) => b.matchingScore - a.matchingScore);
}

async function findKnowledge(
  existingModules: Module[],
  targetRole: string,
  targetSchemaId: string | undefined
): Promise<Knowledge[]> {
  const knowledges: { knowledge: Knowledge, matchingScore: number }[] = [];
  const database = await loadDatabase();
  for (const knowledge of database.knowledges) {
    // Skip those with un-matching target role or un-matching target schema.
    if (knowledge.subjectRole !== targetRole || (
      targetSchemaId !== undefined &&
      knowledge.subjectSchema !== targetSchemaId
    )) {
      continue;
    }

    // Skip those with no similarity at all.
    if (!isWithinScope(knowledge.contextScope, existingModules)) {
      continue;
    }
    // Similarity is currently defined as the matched scope length.
    const similarity = knowledge.contextScope.length;
    knowledges.push({
      knowledge: knowledge,
      matchingScore: similarity
    });
  }
  return knowledges.sort((a, b) => b.matchingScore - a.matchingScore).map((a) => a.knowledge);
}

function findTargetModule(
  modules: Module[],
  targetRole: string,
  targetSchemaId: string | undefined
): Module | undefined {
  return modules.find((module) => {
    return module.role === targetRole && (
      targetSchemaId === undefined || (
        module.role === "verifiedAlgorithm" &&
        (module.module as any).schema === targetSchemaId
      )
    )
  });
}

function hasOverlapModule(modules1: Module[], modules2: Module[]): boolean {
  // In case of no overlapping, similarity can be directly skipped.
  for (const m1 of modules1) {
    for (const m2 of modules2) {
      if (m1.role === m2.role) {
        return true;
      }
    }
  }
  return false;
}

function isWithinScope(modules1: Module[], modules2: Module[]): boolean {
  // Check if all modules in modules1 are within modules2.
  for (const m1 of modules1) {
    const matchedModule = modules2.find((m2) => (
      m2.role === m1.role && (
        (typeof m2.module === "string" && m2.module === m1.module) ||
        (typeof m2.module === "object" && m2.module.id === m1.module)
      )
    ));
    if (matchedModule === undefined) {
      return false;
    }
  }
  return true;
}

async function moduleSimilarity(modules1: Module[], modules2: Module[]): Promise<number> {
  let matchingScore = 0.;
  for (const m1 of modules1) {
    for (const m2 of modules2) {
      if (m1.role === m2.role) {
        if (m1.role === "dataset") {
          return await textSimilarity(
            (m1.module as Dataset).description,
            (m2.module as Dataset).description
          );
        } else if (m1.role === "model") {
          return await textSimilarity(
            (m1.module as Model).description,
            (m2.module as Model).description
          );
        } else if (m1.role === "taskType") {
          return await textSimilarity(
            (m1.module as TaskType).description,
            (m2.module as TaskType).description
          );
        }
        // Ignore other types of modules.
      }
    }
  }
  return matchingScore;
}

async function textSimilarity(text1: string, text2: string): Promise<number> {
  const embed1 = await queryEmbedding(text1);
  const embed2 = await queryEmbedding(text2);
  return (cosineSimilarity(embed1, embed2) + 1.) / 2.;
}

function cosineSimilarity(embed1: Float32Array, embed2: Float32Array): number {
  if (embed1.length !== embed2.length) {
    throw new Error("Embedding length mismatch.");
  }
  let dotProduct = 0.;
  let norm1 = 0.;
  let norm2 = 0.;
  for (let i = 0; i < embed1.length; ++i) {
    dotProduct += embed1[i] * embed2[i];
    norm1 += embed1[i] * embed1[i];
    norm2 += embed2[i] * embed2[i];
  }
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}
