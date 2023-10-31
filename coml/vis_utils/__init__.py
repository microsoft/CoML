from .verifier import VisVerifier


def show_svg(plt):
    """Show a plot as a SVG inline."""
    from io import StringIO

    from IPython.display import SVG, display

    f = StringIO()
    plt.savefig(f, format="svg")
    plt.close()
    display(SVG(f.getvalue()))
