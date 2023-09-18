# CoML

CoML (formerly MLCopilot) assists users in generating practical ML solutions based on historical experiences, streamlining complex ML challenges. Users input specific ML tasks they want to solve, such as classifying emails as spam or not. The system provides suggested solutions, including recommended ML models, data processing methods, and explanations that are easy for humans to understand. [Paper](https://arxiv.org/abs/2304.14979)

![](assets/demo.gif)

(TODO: The demo needs an update.)

### Installation

We currently do not support installation from pypi. Please follow the steps below to install CoML:

1. Clone this repo: `git clone REPO_URL; cd coml`
2. Put assets/coml.db in your home directory: `cp assets/coml.db ~/.coml/coml.db`
3. Copy `coml/.env.template` to `~/.coml/.env` and put your API keys in the file.
3. Install the package via `pip install -e .`.

### Command line utility

CoML can suggest a ML configuration within a specific task, for a specific task. Use the following command line:

```
coml --space <space> --task <task>
```

If you feel uncertain about what to put into `<space>` or `<task>`, see the demo above, or try the interactive usage below:

```
coml --interactive
```

### API Usage

```python
from coml.suggest import suggest

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
