# MLCopilot

MLCopilot is a tool to help you find the best models/hyperparametes for your task. It uses Large Language Models(LLMs) to suggest models and hyperparameters based on your task description and previous experiments.

![](assets/demo.gif)

## Quickstart

1. [Get an OpenAI API Key](#get-an-openai-api-key)
2. [Install requirements](#install-requirements)
3. [Run](#run)

### Get an OpenAI API Key

1. Create an account [here](https://beta.openai.com/signup)
2. Create an API key [here](https://beta.openai.com/account/api-keys)

### Install requirements

0. Clone this repo: `git clone REPO_URL; cd mlcopilot`
1. Put assets/mlcopilot.db in your home directory: `cp assets/mlcopilot.db ~/.mlcopilot/mlcopilot.db`
2. Install Python 3.8 or higher
3. Install: `pip install .`. If you want to develop, use `pip install -e .[dev]` instead.

### Run

Command line: `mlcopilot`


### API Usage

```python
from mlcopilot.suggest import suggest

space = import_space("YOUR_SPACE_ID")
task_desc = "YOUR_TASK_DESCRIPTION_FOR_NEW_TASK"
suggest_configs, knowledge = suggest(space, task_desc)
```



## Citation
If you find this work useful in your method, you can cite the paper as below:

    @article{zhang2023mlcopilot,
        title={MLCopilot: Unleashing the Power of Large Language Models in Solving Machine Learning Tasks},
        author={Zhang, Lei and Zhang, Yuge and Ren, Kan and Li, Dongsheng and Yang, Yuqing},
        journal={arXiv preprint arXiv:2304.14979},
        year={2023}
    }

## License

The entire codebase is under [MIT license](LICENSE).