import { Solution, Algorithm, Dataset, Knowledge, Schema, TaskType, Module } from "./types";

export class Database {
  readonly solutions: Solution[];
  readonly knowledges: Knowledge[];
  readonly schemas: Schema[];
  private algorithmsById: Map<string, Algorithm>;
  private datasetsById: Map<string, Dataset>;
  private taskTypesById: Map<string, TaskType>;
  private schemasById: Map<string, Schema>;

  constructor(
    solutions: Solution[],
    knowledges: Knowledge[],
    schemas: Schema[],
    algorithms: Algorithm[],
    datasets: Dataset[],
    taskTypes: TaskType[]
  ) {
    this.algorithmsById = buildIdIndex(algorithms);
    this.datasetsById = buildIdIndex(datasets);
    this.taskTypesById = buildIdIndex(taskTypes);
    this.schemasById = buildIdIndex(schemas);

    for (const solution of solutions) {
      this.expandModules(solution.modules);
    }

    this.solutions = solutions;
    this.knowledges = knowledges;
    this.schemas = schemas;
  }

  getAlgorithm(id: string): Algorithm {
    return retrieveFromIdIndex(this.algorithmsById, id);
  }

  getDataset(id: string): Dataset {
    return retrieveFromIdIndex(this.datasetsById, id);
  }

  getTaskType(id: string): TaskType {
    return retrieveFromIdIndex(this.taskTypesById, id);
  }

  getSchema(id: string): Schema {
    return retrieveFromIdIndex(this.schemasById, id);
  }

  expandModules(modules: Module[]) {
    for (const module of modules) {
      if (typeof module.module === "string") {
        if (module.role === "algorithm" || module.role === "verifiedAlgorithm") {
          module.module = this.getAlgorithm(module.module);
        } else if (module.role === "dataset") {
          module.module = this.getDataset(module.module);
        } else if (module.role === "taskType") {
          module.module = this.getTaskType(module.module);
        }
      }
    }
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
  const [algorithms, datasets, knowledges, schemas, solutions, taskTypes] = await loadFiles();
  _database = new Database(
    solutions,
    knowledges,
    schemas,
    algorithms,
    datasets,
    taskTypes
  );
  return _database;
}

async function loadFiles(): Promise<[Algorithm[], Dataset[], Knowledge[], Schema[], Solution[], TaskType[]]> {
  if (typeof process === 'object') {
    // This is Node.js
    const fs = await import('fs');
    const url = await import('url');
    const path = await import('path');
    const root = path.join(path.dirname(url.fileURLToPath(import.meta.url)), '..', 'data');
    return await Promise.all([
      fs.promises.readFile(path.join(root, 'algorithms.json')).then((response) => JSON.parse(response.toString())),
      fs.promises.readFile(path.join(root, 'datasets.json')).then((response) => JSON.parse(response.toString())),
      fs.promises.readFile(path.join(root, 'knowledges.json')).then((response) => JSON.parse(response.toString())),
      fs.promises.readFile(path.join(root, 'schemas.json')).then((response) => JSON.parse(response.toString())),
      fs.promises.readFile(path.join(root, 'solutions.json')).then((response) => JSON.parse(response.toString())),
      fs.promises.readFile(path.join(root, 'taskTypes.json')).then((response) => JSON.parse(response.toString())),
    ]);
  } else {
    let root = "./data";
    const selector = document.querySelector('meta[name="data-uri"]');
    if (selector) {
      root = (selector as any).content;
    }
    console.log("Data root: " + root);
    return await Promise.all([
      fetch(`${root}/algorithms.json`).then((response) => response.json()),
      fetch(`${root}/datasets.json`).then((response) => response.json()),
      fetch(`${root}/knowledges.json`).then((response) => response.json()),
      fetch(`${root}/schemas.json`).then((response) => response.json()),
      fetch(`${root}/solutions.json`).then((response) => response.json()),
      fetch(`${root}/taskTypes.json`).then((response) => response.json()),
    ]);
  }
}
