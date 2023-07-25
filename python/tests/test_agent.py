from pathlib import Path

from coml.agent import CoMLAgent, CodingAgent
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


def test_agent():
    load_dotenv(Path("~/.mlcopilot/.env").expanduser())
    llm = ChatOpenAI(temperature=0.5)
    coml_agent = CoMLAgent(llm)
    agent = CodingAgent(llm, coml_agent)

    import numpy as np
    X = np.random.normal(size=(60000, 784))
    y = np.random.normal(size=(60000,))
    agent("Train a ML model with rpart.preproc algorithm for MNIST dataset, a dataset of handwritten digits.", [X, y])
    # agent("Train a ML model for MNIST dataset, a dataset of handwritten digits.", [X, y])

    # agent("Use rpart.preproc algorithm to fit to a MNIST dataset, a dataset of handwritten digits.", [])
    # agent("Get a dataset to pretrain the model.", ["BERT()"])
    # agent("The task is to predict the final price of each home, given 79 explanatory variables describing (almost) every aspect " \
    #     "of residential homes in Ames, Iowa. I want to use xgboost Regressor. How to configure its hyper-parameters?", [])
