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

const levelInverseMapping = new Map<string, number>([
  ["very low", 0],
  ["low", 1],
  ["medium", 2],
  ["high", 3],
  ["very high", 4],
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
  return quantiles[Math.min(quantiles.length - 1, Math.floor((level + 0.5) * quantilesPerLevel))];
}

function configToSemantic(config: any, schema: Schema) {
  let sentence = "";
  schema.parameters.forEach((parameter) => {
    if (parameter.name in config && config[parameter.name]) {
      if (parameter.quantiles) {
        const level = numberToLevel(config[parameter.name], parameter.quantiles, 5);
        sentence += `${parameter.name} is ${levelMapping.get(level)!}. `;
      } else {
        sentence += `${parameter.name} is ${config[parameter.name]}. `;
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

async function moduleAsString(module: Module, roleIndex: number | undefined = undefined): Promise<string> {
  let moduleString = "";
  const moduleRole = roleMapping.get(module.role);
  if (moduleRole) {
    moduleString += capitalize(moduleRole);
    if (roleIndex !== undefined) {
      moduleString += ` ${roleIndex + 1}`;
    }
    if (module.purpose) {
      moduleString += ` (${module.purpose})`;
    }
    moduleString += ": ";
    moduleString += await moduleContentAsString(module);
    moduleString += "\n";
  } else {
    throw new Error(`Unknown module role: ${module.role}`);
  }
  return moduleString;
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
      exampleString += await moduleAsString(input);
    }
    for (let index = 0; index < Math.min(preservedOutputPerInput, example.output.length); index++) {
      exampleString += await moduleAsString(
        example.output[index].candidate,
        example.output.length > 1 ? index : undefined
      );
      if (example.output[index].feedback) {
        exampleString += "Feedback";
        if (example.output.length > 1) {
          exampleString += ` ${index + 1}`;
        }
        exampleString += ": ";
        exampleString += example.output[index].feedback;
        exampleString += "\n";
      }
    }
    if (joinedExamples.length + exampleString.length > maxCharacters) {
      break;
    }
    joinedExamples += exampleString + "\n";
  }
  return "Here are several examples:\n\n" + joinedExamples;
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
): Promise<string> {
  const targetRoleName = roleMapping.get(targetRole)!;
  let prefix = `Your task is to find a suitable ${targetRoleName} for a machine learning pipeline.\n`;
  if (existingModules.length > 0) {
    prefix += `The pipeline has already been partially constructed with the following components:\n`;
    for (const module of existingModules) {
      prefix += await moduleAsString(module);
    }
  }
  prefix += "\n";

  let suffix = "\n";
  suffix += `Your response should contain ${preservedOutputPerInput} candidate ${targetRoleName}s, `
  suffix += "where each candidate is written in one line, and the most confident one is put upfront. "
  suffix += "You should also strictly follow the following format:\n";
  suffix += `${capitalize(targetRoleName)} [`;
  suffix += Array.from(Array(preservedOutputPerInput).keys()).map(i => (i + 1).toString()).join("|") + "]: ";
  if (targetSchema) {
    suffix += `[parameterName] is [${Array.from(levelMapping.values()).join("|")}].\n\n`;
    suffix += "The available parameters are as follows:\n"
    for (const parameter of targetSchema.parameters) {
      suffix += `- ${parameter.name}: ${parameter.dtype ? parameter.dtype + ". " : ""}`;
      if (parameter.categorical) {
        suffix += "Choosing from " + parameter.choices!.join(", ") + ". ";
      } else {
        suffix += `Range from ${parameter.low!} to ${parameter.high!}. `;
        if (parameter.logDistributed) {
          suffix += "Log distributed. ";
        }
      }
      if (parameter.condition) {
        for (const condition of parameter.condition) {
          if (condition.match) {
            for (const [key, value] of Object.entries(condition.match)) {
              suffix += `Only when ${key} is ${value}. `;
            }
          }
        }
      }
      suffix = suffix.trimEnd() + "\n";
    }
  } else if (targetRole === "algorithm") {
    suffix += "[JSON string].\n";
  } else {
    suffix += "[string].\n";
  }
  suffix += "\n"
  if (existingModules.length > 0) {
    suffix += `Please take into consideration that the following components have already existed on the pipeline:\n`;
    for (const module of existingModules) {
      suffix += await moduleAsString(module);
    }
    suffix += "\n"
  }

  const knowledgePart = generateKnowledgeMessage(knowledge, knowledgeMaxCharacters);
  // Fill the rest.
  const examplePart = await generateExampleMessage(
    examples, targetRole,
    maxCharacters - knowledgePart.length - prefix.length - suffix.length,
    preservedOutputPerInput
  );
  return [prefix, examplePart, knowledgePart, suffix].join("\n");
}

export function parseResponse(
  response: string,
  targetRole: string,
  targetSchema: Schema | undefined
): Module[] {
  const targetRoleName = roleMapping.get(targetRole)!;
  const findCandidateRegex = new RegExp(
    `${capitalize(targetRoleName)} (\\d+): (.*)`, "g"
  );
  const modules: Module[] = [];
  if (targetSchema) {
    for (const line of response.matchAll(findCandidateRegex)) {
      const candidateAlgo: any = {};
      for (const item of line[2].matchAll(/(.*?) is (.*?)\. ?/g)) {
        const parameterName = item[1];
        const parameterValue = item[2];
        const parameter = targetSchema.parameters.find(p => p.name === parameterName);
        if (!parameter) {
          console.error(`Unknown parameter name: ${parameterName}. Skip.`);
          continue;
        }
        if (parameter.categorical) {
          if (!parameter.choices!.includes(parameterValue)) {
            console.error(`Unknown parameter value: ${parameterValue}. Skip.`);
            continue;
          }
          candidateAlgo[parameterName] = parameterValue;
        } else if (parameter.quantiles) {
          const level = levelInverseMapping.get(parameterValue);
          if (level === undefined) {
            console.error(`Unknown parameter value: ${parameterValue}. Skip.`);
            continue;
          }
          const value = levelToNumber(level, parameter.quantiles, 5);
          candidateAlgo[parameterName] = value;
        } else {
          const value = parseFloat(parameterValue);
          if (isNaN(value)) {
            console.error(`Unknown parameter value: ${parameterValue}. Skip.`);
            continue;
          } else if (value < parameter.low! || value > parameter.high!) {
            console.error(`Parameter value out of range: ${parameterValue}. Skip.`);
            continue;
          }
          candidateAlgo[parameterName] = value;
        }
      }
      modules.push({
        role: targetRole as any,
        module: {
          schema: targetSchema.id,
          config: candidateAlgo
        }
      });
    }
  } else {
    for (const line of response.matchAll(findCandidateRegex)) {
      if (targetRole === "algorithm") {
        modules.push({
          role: targetRole as any,
          module: {
            config: JSON.parse(line[2])
          }
        });
      } else if (targetRole === "solutionSummary") {
        modules.push({
          role: targetRole as any,
          module: {
            summary: line[2]
          }
        });
      } else {
        modules.push({
          role: targetRole as any,
          module: {
            name: "UNKNOWN",
            description: line[2]
          }
        });
      }
    }
  }
  return modules;
}
