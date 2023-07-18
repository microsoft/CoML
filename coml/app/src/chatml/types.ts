export interface SolutionSummary {
  id?: string;
  summary: string;
}

export interface Dataset {
  id?: string;
  name: string;
  description: string;
}

export interface TaskType {
  id?: string;
  name: string;
  description: string;
}

export interface Model {
  id?: string;
  name: string;
  description: string;
}

export interface Algorithm {
  id?: string;
  config: any;
}

export interface VerifiedAlgorithm {
  id?: string;
  schema: string;
  config: any;
}

export interface Schema {
  id?: string;
  description: string;
  parameters: Parameter[];
}

export interface Parameter {
  name: string;
  dtype: "int" | "float" | "str" | "bool" | undefined;
  categorical: boolean;
  choices: string[];
  low?: number;
  high?: number;
  logDistributed?: boolean;
  condition?: Condition[];
  quantiles?: number[];
}

export interface Condition {
  match?: any;
}

export interface Solution {
  id: string;
  modules: Module[];
  metrics: number | Metric[] | undefined;
  source: "hpob" | "huggingface" | "kaggle";
}

export interface Knowledge {
  id: string;
  contextScope: Module[]; // AND
  subjectRole: string;
  subjectSchema?: string;
  knowledge: string;
}

export interface Metric {
  dataset: Dataset;
  metric: number;
  extra: string;
  split?: string;
  protocol?: string;
}

export interface Module {
  role: "dataset" | "taskType" | "model" | "algorithm" | "verifiedAlgorithm" | "solutionSummary";
  purpose?: string;
  module: Dataset | SolutionSummary | TaskType | Model | Algorithm | VerifiedAlgorithm | string;
}
