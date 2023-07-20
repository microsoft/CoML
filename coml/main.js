import { suggestMachineLearningModule } from "./dist/index";

const suggest = suggestMachineLearningModule([
    {
        role: "dataset",
        module: {
          name: "MNIST",
          description: "A dataset of handwritten digits",
        }
    },
], "verifiedAlgorithm", "rpart-preproc-4796");
console.log(suggest);
