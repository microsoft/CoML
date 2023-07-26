def ipython_available():
    # https://stackoverflow.com/q/15411967
    try:
        get_ipython()  # type: ignore
        return True
    except NameError:
        return False


def create_new_cell(contents):
    # https://stackoverflow.com/q/54987129
    shell = get_ipython()  # type: ignore
    payload = dict(
        source="set_next_input",
        text=contents,
        replace=False,
    )
    shell.payload_manager.write_payload(payload, single=False)
