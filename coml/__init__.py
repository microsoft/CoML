from pathlib import Path

from ._version import __version__
from .core import CoMLAgent
from .prompt_utils import describe_variable, filter_variables


def load_ipython_extension(ipython):
    from IPython.core.display import Javascript
    from IPython.display import display

    from .magics import CoMLMagics

    display(
        Javascript((Path(__file__).parent / "js" / "nbclassic_init.js").read_text())
    )

    ipython.register_magics(CoMLMagics)

    print(f"CoML {__version__} loaded.")


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "coml"}]
