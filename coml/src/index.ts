import { ChatOpenAI } from "langchain/chat_models/openai";
import { HumanMessage, SystemMessage, BaseMessage, AIMessage, FunctionMessage } from "langchain/schema";

import { loadDatabase } from "./database";
import { openAIApiKey } from "./apiKey";
import { queryEmbedding, preprocessEmbeddings } from "./embedding";
import { Example, generateHumanMessage, parseResponse } from "./prompt";
import { Module, Solution, VerifiedAlgorithm, Dataset, Model, TaskType, Metric, Knowledge } from "./types";

export async function getFunctionDescription() {
  const validSchemas = (await loadDatabase()).schemas.map(schema => `- ${schema.id}: ${schema.description}`).join("\n");
  return {
    name: "suggestMachineLearningModule",
    description: "Get recommendations of a machine learning module given existing modules on the pipeline. " +
      "A machine learning pipeline consists of multiple modules playing different roles. " +
      "For example, a pipeline can consist of a dataset, a task type, a model, and an algorithm. " +
      "This function recommends a module of the target role given existing modules on the pipeline.",
    parameters: {
      type: "object",
      properties: {
        existingModules: {
          type: "array",
          description: "Existing modules on the pipeline. It can be a dataset, a selected ML model, " +
            "a certain task type, an overview of the whole ML solution, or an algorithm configuration. " +
            "Existing modules MUST BE NOT EMPTY.",
          items: {
            type: "object",
            properties: {
              role: {
                type: "string",
                enum: ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"],
                description: "The role of the module within the pipeline.\n" +
                  "- dataset: Data used for training or testing.\n" +
                  "- taskType: The type of the machine learning task, e.g., image classification.\n" +
                  "- model: A program that fits onto the training data and make predictions.\n" +
                  "- algorithm: Any ML component that can be expressed with a configuration, e.g., training hyper-parameters, data-preprocessing steps, etc.\n" +
                  "- verifiedAlgorithm: An algorithm that strictly follows a schema and thus directly runnable.\n" +
                  "- solutionSummary: An overview of the entire machine learning pipeline/solution."
              },
              purpose: {
                type: "string",
                description: "Why this module is used on the pipeline."
              },
              module: {
                type: "object",
                properties: {
                  id: {
                    type: "string",
                    description: "ID of the module in the database. Do not use this field if you are not sure."
                  },
                  name: {
                    type: "string",
                    description: "Name of the dataset / model / task type. Only use this field when role is dataset/model/taskType."
                  },
                  description: {
                    type: "string",
                    description: "Description of the dataset / model / task type. Only use this field when role is dataset/model/taskType."
                  },
                  summary: {
                    type: "string",
                    description: "Summary of the solution. Only use this field when role is solutionSummary."
                  },
                  config: {
                    type: "object",
                    description: "Configuration of the algorithm. Only use this field when role is algorithm/verifiedAlgorithm."
                  },
                  schema: {
                    type: "string",
                    description: "Schema ID of the algorithm. Only use this field when role is verifiedAlgorithm."
                  }
                }
              }
            },
            required: ["role", "module"]
          },
          minItems: 1
        },
        targetRole: {
          type: "string",
          description: "The role of the module to be recommended.",
          enum: ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"]
        },
        targetSchemaId: {
          type: "string",
          description: "This field should be used together with targetRole = verifiedAlgorithm." +
            "The function will return an algorithm that is valid for the specified schema. " +
            "Valid schema IDs and descriptions are:\n" + validSchemas
        }
      },
      required: ["existingModules", "targetRole"]
    },
  };
}

export async function chatWithGPT(messages: BaseMessage[]): Promise<AIMessage> {
  const model = new ChatOpenAI({
    openAIApiKey: openAIApiKey,
    temperature: 0.5,
    topP: 1,
    modelName: "gpt-3.5-turbo"
  });

  const functionDescription = await getFunctionDescription();

  return await model.call([
    new SystemMessage(
      "Your task as a helpful assistant is to assist users in generating machine learning pipelines." +
      "Your final goal is to generate runnable code snippets that can be directly adopted by users. " +
      "To accomplish this, please utilize the `suggestMachineLearningModule` function as an intermediate step. " +
      "When using this function, your should first identify the datasets, models, task types, " +
      "and other existing components specified by the users, if provided. " +
      "Additionally, you should determine the specific type of module that users are interested in."
    ),
    ...messages
  ], {
    functions: [functionDescription],
    function_call: messages[messages.length - 1] instanceof FunctionMessage ? "none" : "auto"
  });
}

export async function prepareCache() {
  const database = await loadDatabase();
  const texts: string[] = [];
  for (const solution of database.solutions) {
    for (const module of solution.modules) {
      if (module.role === "dataset") {
        texts.push((module.module as Dataset).description);
      } else if (module.role === "model") {
        texts.push((module.module as Model).description);
      } else if (module.role === "taskType") {
        texts.push((module.module as TaskType).description);
      }
    }
  }
  await preprocessEmbeddings(Array.from(new Set<string>(texts)));
}

export async function suggestMachineLearningModule(
  existingModules: Module[],
  targetRole: string,
  targetSchemaId: string | undefined = undefined
): Promise<Module[]> {
  const logTheme = "color: #0078d4";
  const promptTheme = "color: #038387";
  const responseTheme = "color: #ca5010";

  const database = await loadDatabase();
  database.expandModules(existingModules);
  console.log('%cModules received.', logTheme);
  console.log(existingModules);
  const examples = await findExamples(existingModules, targetRole, targetSchemaId);
  console.log('%cExamples found.', logTheme);
  console.log((await examples).slice(0, 10));
  const knowledges = await findKnowledge(existingModules, targetRole, targetSchemaId);
  console.log('%cKnowledge found.', logTheme);
  console.log((await knowledges).slice(0, 10));

  const targetSchema = targetSchemaId ? database.getSchema(targetSchemaId) : undefined;
  const comlPrompt = await generateHumanMessage(examples, knowledges, existingModules, targetRole, targetSchema);
  console.log("%cPrompt generated:\n\n" + comlPrompt, promptTheme);

  const model = new ChatOpenAI({
    openAIApiKey: openAIApiKey,
    temperature: 0.,
    topP: 1,
    modelName: "gpt-3.5-turbo"
  });
  const response = await model.call([
    new SystemMessage("You are a data scientist who is good at solving machine learning problems."),
    new HumanMessage(comlPrompt)
  ]);
  console.log("%cResponse received:\n\n" + response.content, responseTheme);

  const parsedResponse = parseResponse(response.content, targetRole, targetSchema);
  console.log("%cResponse parsed:", logTheme);
  console.log(parsedResponse);
  return parsedResponse;
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
          matchingScore += await textSimilarity(
            (m1.module as Dataset).description,
            (m2.module as Dataset).description
          );
        } else if (m1.role === "model") {
          matchingScore += await textSimilarity(
            (m1.module as Model).description,
            (m2.module as Model).description
          );
        } else if (m1.role === "taskType") {
          matchingScore += await textSimilarity(
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
