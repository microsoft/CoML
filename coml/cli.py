from typing import Optional

import click

from .orm import database_proxy


@click.group(invoke_without_command=True)
@click.option("--space", help="Space ID.")
@click.option("--task", help="Task description.")
@click.option("--interactive", help="Interactive mode.", is_flag=True)
@click.pass_context
def main(
    ctx: click.Context,
    space: Optional[str] = None,
    task: Optional[str] = None,
    interactive: bool = False,
) -> None:
    if ctx.invoked_subcommand is None:
        if ctx.params["interactive"]:
            from .suggest import suggest_interactive

            suggest_interactive()
            database_proxy.close()
        else:
            if ctx.params["space"] is None or ctx.params["task"] is None:
                print("Please specify space ID and a task description.")
                return
            from .space import import_space
            from .suggest import print_suggested_configs, suggest

            results = suggest(import_space(ctx.params["space"]), ctx.params["task"])
            print_suggested_configs(*results)
            database_proxy.close()


@main.command()
@click.argument("space", nargs=1)
@click.argument("history", nargs=1)
@click.argument("task_desc", nargs=1)
@click.option("--space-desc", help="Space description path (optional).")
@click.option("--no-knowledge", help="Do not generate knowledge.", is_flag=True)
def create(
    space: str,
    history: str,
    task_desc: str,
    space_desc: str = None,
    no_knowledge: bool = False,
) -> None:
    """
    Create a space from history csv file and task description json file.

    Parameters
    ----------
    space: str
        The ID of the space to identify the space.
    history: str
        The path to the history of configurations. A csv file, format see `coml.experience.ingest_experience`.
    task_desc: str
        The JSON path to the task description. A json file, format see `coml.experience.ingest_experience`.
    space_desc: str
        The text path to the space description. Optional.
    no_knowledge: bool
        Whether to generate knowledge from history.

    Returns
    -------
    None
    """
    from .space import create_space

    create_space(space, history, task_desc, space_desc, no_knowledge)
    database_proxy.close()


@main.command()
def list() -> None:
    from .space import print_space

    print_space()
    database_proxy.close()


@main.command()
@click.argument("space", nargs=1)
def delete(space: str) -> None:
    from .space import delete_space

    delete_space(space)
    database_proxy.close()
