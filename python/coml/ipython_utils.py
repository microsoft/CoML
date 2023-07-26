def ipython_available():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        get_ipython()  # type: ignore
        return True
    except NameError:
        return False
