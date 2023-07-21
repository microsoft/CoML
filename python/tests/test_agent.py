from pathlib import Path

from coml.agent import CoMLAgent
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


def test_agent():
    load_dotenv(Path("~/.mlcopilot/.env").expanduser())
    agent = CoMLAgent(ChatOpenAI(temperature=0.5))

    import numpy as np
    X = np.random.normal(size=(60000, 784))
    y = np.random.normal(size=(60000,))
    # agent("Train a ML model with rpart.preproc algorithm for MNIST dataset, a dataset of handwritten digits.", [X, y])
    agent("Train a ML model for MNIST dataset, a dataset of handwritten digits.", [X, y])
