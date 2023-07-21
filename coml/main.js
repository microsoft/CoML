import { suggestMachineLearningModule, getFunctionDescription } from "./dist/index";

if (process.argv.length < 3) {
  console.log("Usage: node main.js <suggestMachineLearningModule|getFunctionDescription>");
  process.exit(1);
}

const logIdentifier = "<|coml_nodejs|>";

const command = process.argv[2];
if (command === "suggestMachineLearningModule") {
  if (process.argv.length < 5) {
    console.log("Usage: node main.js suggestMachineLearningModule <targetRole> <targetSchemaId>");
    process.exit(1);
  }
  const existingModules = JSON.parse(process.argv[3]);
  const targetRole = process.argv[4];
  const targetSchemaId = process.argv.length > 5 ? process.argv[5] : undefined;
  const suggest = await suggestMachineLearningModule(existingModules, targetRole, targetSchemaId);
  console.log(logIdentifier);
  console.log(JSON.stringify(suggest, null, 2));
} else if (command === "getFunctionDescription") {
  const functionDescription = await getFunctionDescription();
  console.log(logIdentifier);
  console.log(JSON.stringify(functionDescription, null, 2));
} else {
  console.log(`Unknown command: ${command}`);
  process.exit(1);
}
