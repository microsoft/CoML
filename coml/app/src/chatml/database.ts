import { Solution, Algorithm, Dataset, Knowledge, Schema, TaskType } from "./types";

export class Database {
  readonly solutions: Solution[];
  readonly knowledges: Knowledge[];
  readonly schemas: Schema[];

  constructor(solutions: Solution[], knowledges: Knowledge[], schemas: Schema[]) {
    this.solutions = solutions;
    this.knowledges = knowledges;
    this.schemas = schemas;
  }
}

let _database: Database | undefined = undefined;

interface HasId {
  id?: string;
}

function buildIdIndex<T extends HasId>(array: T[]): Map<string, T> {
  const index = new Map<string, T>();
  for (const item of array) {
    if (!item.id) {
      throw new Error("Item does not have an id.");
    }
    if (index.has(item.id)) {
      throw new Error(`Duplicate id: ${item.id}`);
    }
    index.set(item.id, item);
  }
  return index;
}

function retrieveFromIdIndex<T extends HasId>(index: Map<string, T>, id: string): T {
  const item = index.get(id);
  if (item === undefined) {
    throw new Error(`Id not found: ${id}`);
  }
  return item;
}


export async function loadDatabase(): Promise<Database> {
  if (_database !== undefined) {
    return _database;
  }
  const responses: [Algorithm[], Dataset[], Knowledge[], Schema[], Solution[], TaskType[]] = await Promise.all([
    fetch("./data/algorithms.json").then((response) => response.json()),
    fetch("./data/datasets.json").then((response) => response.json()),
    fetch("./data/knowledges.json").then((response) => response.json()),
    fetch("./data/schemas.json").then((response) => response.json()),
    fetch("./data/solutions.json").then((response) => response.json()),
    fetch("./data/taskTypes.json").then((response) => response.json()),
  ]);
  const [algorithms, datasets, knowledges, schemas, solutions, taskTypes] = responses;
  const algorithmIndex = buildIdIndex(algorithms);
  const datasetIndex = buildIdIndex(datasets);
  const taskTypeIndex = buildIdIndex(taskTypes);
  for (const solution of solutions) {
    for (const module of solution.modules) {
      if (typeof module.module === "string") {
        // Expand nested modules
        if (module.role === "algorithm" || module.role === "verifiedAlgorithm") {
          module.module = retrieveFromIdIndex(algorithmIndex, module.module);
        } else if (module.role === "dataset") {
          module.module = retrieveFromIdIndex(datasetIndex, module.module);
        } else if (module.role === "taskType") {
          module.module = retrieveFromIdIndex(taskTypeIndex, module.module);
        }
      }
    }
  }
  _database = new Database(solutions, knowledges, schemas);
  return _database;
}
