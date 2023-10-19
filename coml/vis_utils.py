from io import StringIO

import matplotlib.figure
from IPython.display import SVG, display


def show_svg(plt: matplotlib.figure.Figure):
    """Show a plot as a SVG inline."""
    f = StringIO()
    plt.savefig(f, format="svg")
    plt.close()
    display(SVG(f.getvalue()))
