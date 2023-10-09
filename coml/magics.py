import warnings
from typing import Any

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic, no_var_expand
from IPython.display import Code, display
import ipywidgets as widgets
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from .core import ChatDataAgent
from .ipython_utils import (
    get_ipython_history, run_code_in_next_cell, insert_cell_below, get_last_cell,
    parse_cell_outputs
)
from .prompt_utils import (
    GenerateContext, FixContext,
    filter_variables, describe_variable,
    InteractionIncomplete
)


@magics_class
class ChatDataMagics(Magics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import dotenv
        dotenv.load_dotenv()
        llm = ChatOpenAI(temperature=0., model="gpt-3.5-turbo-16k")
        self.agent = ChatDataAgent(llm)

    def _get_variable_context(self) -> dict[str, Any]:
        assert self.shell is not None
        return {key: describe_variable(value) for key, value in filter_variables(self.shell.user_ns).items()}

    def _get_code_context(self) -> list[str]:
        assert self.shell is not None
        return get_ipython_history(self.shell)

    def _post_generation(self, code: str, context: GenerateContext | FixContext) -> None:
        def run_button_on_click(b):
            run_code_in_next_cell("%%runit\n" + code, context)

        def edit_button_on_click(b):
            insert_cell_below(code, context)

        def explain_button_on_click(b):
            run_code_in_next_cell("%%explainit\n" + code)

        run_button = widgets.Button(description="👍 Run it!", layout=widgets.Layout(width="33%"))
        edit_button = widgets.Button(description="🤔 Let me edit.", layout=widgets.Layout(width="33%"))
        explain_button = widgets.Button(description="🧐 Explain it.", layout=widgets.Layout(width="33%"))
        run_button.on_click(run_button_on_click)
        edit_button.on_click(edit_button_on_click)
        explain_button.on_click(explain_button_on_click)

        combined = widgets.HBox([run_button, edit_button, explain_button])
        display(Code(code, language="python"))
        display(combined)

    def _fix_context_from_cell(self, source: str, **kwargs: Any) -> FixContext:
        return FixContext(
            variables=self._get_variable_context(),
            codes=self._get_code_context(),
            request=None,
            first_attempt=source,
            interactions=[
                InteractionIncomplete(**kwargs)
            ]
        )

    @no_var_expand
    @line_cell_magic
    def helpme(self, line, cell=None):
        request: str = line
        if cell is not None:
            request += "\n" + cell
        generate_context = self.agent.generate_code(
            request.strip(),
            self._get_variable_context(),
            self._get_code_context()
        )
        return self._post_generation(generate_context["answer"], generate_context)

    @no_var_expand
    @line_magic
    def inspireme(self, line):
        if line:
            warnings.warn(r"The argument of %inspireme is ignored.")
        suggestions = self.agent.suggest(self._get_code_context())

        def run_button_on_click(b):
            run_code_in_next_cell(r"%helpme " + b.description)

        buttons = [widgets.Button(description=s, layout=widgets.Layout(width="100%")) for s in suggestions]
        for button in buttons:
            button.on_click(run_button_on_click)
        display(widgets.VBox(buttons))

    @no_var_expand
    @line_magic
    def fixit(self, line):
        hint: str | None = line.strip()
        if not hint:
            hint = None

        target_cell = get_last_cell()
        if target_cell is None:
            warnings.warn("No cell to fix!")
            return
        if target_cell["cell_type"] != "code":
            warnings.warn("Only code cells can be fixed.")
            return

        error, output = parse_cell_outputs(target_cell["outputs"])
        if "chatdata" in target_cell["metadata"]:
            context = target_cell["metadata"]["chatdata"]
        else:
            # Last cell is created by user.
            print("This cell is not created by chatdata. Still trying to fix it though.")
            context = FixContext(
                variables=self._get_variable_context(),
                codes=self._get_code_context(),
                request=None,
                first_attempt=target_cell["source"],
                interactions=[
                    InteractionIncomplete(error=error, output=output, hint=hint)
                ]
            )

        fix_context = self.agent.fix_code(error, output, hint, context)
        if fix_context is None:
            return
        assert "code" in fix_context["interactions"][-1]
        return self._post_generation(fix_context["interactions"][-1]["code"], fix_context)

    @no_var_expand
    @cell_magic
    def explainit(self, line, cell):
        if line:
            warnings.warn(r"The argument of %%explainit is ignored.")
        explanation = self.agent.explain(cell)
        display(Code(explanation, language="markdown"))

    @no_var_expand
    @cell_magic
    def runit(self, line, cell):
        if line:
            warnings.warn(r"The argument of %runit is ignored.")
        assert self.shell is not None
        output = None
        try:
            output = self.shell.run_cell(cell)
            return output.result
        finally:
            def like_button_on_click(b):
                print("Thanks for your feedback! 🤗")

            def fix_button_on_click(b):
                run_code_in_next_cell(r"%fixit")

            def fix_with_comment_button_on_click(b):
                insert_cell_below(r"%fixit <describe the problem here>")

            like_button = widgets.Button(description="🤗 Looks good!", layout=widgets.Layout(width="33%"))
            retry_button = widgets.Button(description="🤬 Try again!", layout=widgets.Layout(width="33%"))
            comment_button = widgets.Button(description="🤯 I'll show you what's wrong.", layout=widgets.Layout(width="33%"))
            like_button.on_click(like_button_on_click)
            retry_button.on_click(fix_button_on_click)
            comment_button.on_click(fix_with_comment_button_on_click)

            combined = widgets.HBox([like_button, retry_button, comment_button])
            display(combined)
