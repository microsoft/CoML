import { loadDatabase } from "./database";
import {
  Module, Solution, Algorithm, VerifiedAlgorithm, Dataset,
  Model, TaskType, Metric, Knowledge, Schema, SolutionSummary
} from "./types";

const roleMapping = new Map<string, string>([
  ["dataset", "dataset"],
  ["algorithm", "algorithm"],
  ["taskType", "task type"],
  ["model", "model"],
  ["verifiedAlgorithm", "algorithm"],
  ["solutionSummary", "context"],
]);

const levelMapping = new Map<number, string>([
  [0, "very low"],
  [1, "low"],
  [2, "medium"],
  [3, "high"],
  [4, "very high"],
]);

export interface Example {
  input: Module[];
  output: {
    candidate: Module;
    metric: number;
    feedback?: string;
  }[];
  matchingScore: number;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function numberToLevel(value: number, quantiles: number[], numLevels: number): number {
  const quantilesPerLevel = (quantiles.length - 1) / numLevels;
  for (let i = 0; i < numLevels; i++) {
    if (value <= quantiles[Math.min(quantiles.length - 1, Math.floor((i + 1) * quantilesPerLevel))]) {
      return i;
    }
  }
  return numLevels - 1;
}

function levelToNumber(level: number, quantiles: number[], numLevels: number) {
  const quantilesPerLevel = (quantiles.length - 1) / numLevels;
  return quantiles[Math.min(quantiles.length, Math.floor((level + 0.5) * quantilesPerLevel))];
}

function configToSemantic(config: any, schema: Schema) {
  let sentence = "";
  schema.parameters.forEach((parameter) => {
    if (parameter.name in config && config[parameter.name]) {
      if (parameter.categorical) {
        sentence += `${parameter.name} is ${config[parameter.name]}. `;
      } else {
        const level = numberToLevel(config[parameter.name], parameter.quantiles!, 5);
        sentence += `${parameter.name} is ${levelMapping.get(level)!}. `;
      }
    }
  });
  return sentence.trim();
}

async function moduleContentAsString(module: Module): Promise<string> {
  if (module.role === "dataset") {
    return (module.module as Dataset).description;
  } else if (module.role === "taskType") {
    return (module.module as TaskType).description;
  } else if (module.role === "model") {
    return (module.module as Model).description;
  } else if (module.role === "algorithm") {
    return JSON.stringify((module.module as Algorithm).config);
  } else if (module.role === "verifiedAlgorithm") {
    const database = await loadDatabase();
    const schema = database.getSchema((module.module as VerifiedAlgorithm).schema);
    return configToSemantic((module.module as VerifiedAlgorithm).config, schema);
  } else if (module.role === "solutionSummary") {
    return (module.module as SolutionSummary).summary;
  } else {
    throw new Error(`Unknown module role: ${module.role}`);
  }
}

async function generateExampleMessage(
  examples: Example[],
  targetRole: string,
  maxCharacters: number,
  preservedOutputPerInput: number,
): Promise<string> {
  let joinedExamples = "";
  for (const example of examples) {
    let exampleString = "";
    for (const input of example.input) {
      const inputRole = roleMapping.get(input.role);
      if (inputRole) {
        exampleString += capitalize(inputRole);
        if (input.purpose) {
          exampleString += ` (${input.purpose})`;
        }
        exampleString += ": ";
        exampleString += await moduleContentAsString(input);
        exampleString += "\n";
      } else {
        throw new Error(`Unknown module role: ${input.role}`);
      }
    }
    for (let index = 0; index < Math.min(preservedOutputPerInput, example.output.length); index++) {
      const outputRole = roleMapping.get(targetRole);
      if (outputRole) {
        exampleString += capitalize(outputRole);
        if (example.output.length > 1) {
          exampleString += ` ${index + 1}`;
        }
        exampleString += ": ";
        exampleString += await moduleContentAsString(example.output[index].candidate);
        exampleString += "\n";
        if (example.output[index].feedback) {
          exampleString += "Feedback";
          if (example.output.length > 1) {
            exampleString += ` ${index + 1}`;
          }
          exampleString += ": ";
          exampleString += example.output[index].feedback;
          exampleString += "\n";
        }
      } else {
        throw new Error(`Unknown module role: ${targetRole}`);
      }
    }
    if (joinedExamples.length + exampleString.length > maxCharacters) {
      break;
    }
    joinedExamples += exampleString + "\n";
  }
  return joinedExamples;
}

function generateKnowledgeMessage(knowledge: Knowledge[], maxCharacters: number) {
  const knowledgeMessages = knowledge.map((k, index) => `${index + 1}. ${k.knowledge}`);
  let joinedMessage = "";
  for (const message of knowledgeMessages) {
    if (joinedMessage.length + message.length > maxCharacters) {
      break;
    }
    joinedMessage += message + "\n";
  }
  if (joinedMessage) {
    joinedMessage = "Here are several instructions that might be helpful to you:\n" + joinedMessage;
  }
  return joinedMessage;
}

export async function generateHumanMessage(
  examples: Example[],
  knowledge: Knowledge[],
  existingModules: Module[],
  targetRole: string,
  targetSchema: Schema | undefined,
  preservedOutputPerInput: number = 3,
  maxCharacters: number = 10000,
  knowledgeMaxCharacters: number = 1000,
) {
  const knowledgePart = generateKnowledgeMessage(knowledge, knowledgeMaxCharacters);
  const examplePart = await generateExampleMessage(
    examples, targetRole,
    maxCharacters - knowledgePart.length,
    preservedOutputPerInput
  );
  return examplePart + "\n" + knowledgePart;
}
