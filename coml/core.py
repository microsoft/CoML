from __future__ import annotations

import copy
import re
import warnings
from typing import Any, cast

import colorama
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .prompt_utils import (
    CHECK_INSTRUCTION,
    EXPLAIN_INSTRUCTION,
    FIX_INSTRUCTION,
    GENERATE_INSTRUCTION,
    SANITY_CHECK_INSTRUCTION,
    SUGGEST_INSTRUCTION,
    FixContext,
    GenerateContext,
    GenerateContextIncomplete,
    Interaction,
    InteractionIncomplete,
    cached_fix_fewshots,
    cached_generate_fewshots,
    render_check_context,
    render_fix_context,
    render_generate_context,
    render_ipython_cells,
    render_sanity_check_context,
)
from .vis_utils import VisVerifier

_debug_mode: bool = False


def debug_messages(*messages: BaseMessage) -> None:
    if not _debug_mode:
        return
    for message in messages:
        if isinstance(message, SystemMessage):
            print(colorama.Fore.BLUE + message.content + colorama.Fore.RESET + "\n")
        elif isinstance(message, HumanMessage):
            print(colorama.Fore.GREEN + message.content + colorama.Fore.RESET + "\n")
        elif isinstance(message, AIMessage):
            print(colorama.Fore.MAGENTA + message.content + colorama.Fore.RESET + "\n")


def parse_fix(response: str) -> tuple[str, str, str]:
    match = re.search(
        r"Here is a line-by-line explanation of the code:\s([\s\S]+)\s"
        r"Observe what is wrong with the code:\s([\s\S]+)\s"
        r"The fixed code:\s+```.*\n([\s\S]+?)\n```",
        response,
    )
    match_wo_code = re.search(
        r"Here is a line-by-line explanation of the code:\s([\s\S]+)\s"
        r"Observe what is wrong with the code:\s([\s\S]+)\s",
        response,
    )
    if match is not None:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    elif match_wo_code is not None:
        return match_wo_code.group(1).strip(), match_wo_code.group(2).strip(), ""
    else:
        return "", "", parse_code(response)


def parse_code(response: str) -> str:
    patterns = [
        r"```.*\n([\s\S]+?)\n```",
        r"```.*\n([\s\S]+?)```",
        r"```([\s\S]+?)```",
        r"```.*\n([\s\S]+)",
        r"```([\s\S]+)\n```",
        r"```([\s\S]+)```",
        r"(.*)",
    ]
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, response)
        if match is not None:
            if index > 0:
                warnings.warn(
                    f"Unable to parse the code perfectly from response. "
                    f"Using pattern {pattern}."
                )
            return match.group(1)
    # Give up. Return whole response.
    warnings.warn("Unable to parse the code from response.")
    return response


class CoMLAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _fix_context_from_any_context(
        self, context: GenerateContext | FixContext, **kwargs: Any
    ) -> FixContext:
        if "answer" in context:
            return FixContext(
                variables=context["variables"],
                codes=context["codes"],
                request=context["request"],
                first_attempt=context["answer"],
                interactions=[InteractionIncomplete(**kwargs)],
            )
        else:
            context = context.copy()
            context["interactions"].append(InteractionIncomplete(**kwargs))
            return context

    def generate_code(
        self, request: str, variable_descriptions: dict[str, str], codes: list[str]
    ) -> GenerateContext:
        fewshots = cached_generate_fewshots()
        messages: list[BaseMessage] = [
            SystemMessage(content=GENERATE_INSTRUCTION),
        ]
        for shot in fewshots:
            question, answer = render_generate_context(shot)
            messages.append(HumanMessage(content=question))
            if answer is not None:
                messages.append(AIMessage(content=answer))
        context = GenerateContextIncomplete(
            variables=variable_descriptions, codes=codes, request=request
        )
        question, _ = render_generate_context(context)
        messages.append(HumanMessage(content=question))
        debug_messages(*messages)

        response = self.llm(messages)
        debug_messages(response)
        code = parse_code(response.content)
        return {**context, "answer": code}

    def fix_code(
        self,
        error: str | None,
        output: str | None,
        hint: str | None,
        prev_context: GenerateContext | FixContext,
    ) -> FixContext | None:
        fewshots = cached_fix_fewshots()
        messages: list[BaseMessage] = [
            SystemMessage(content=FIX_INSTRUCTION),
        ]
        context = self._fix_context_from_any_context(
            prev_context, error=error, output=output, hint=hint
        )
        for shot in fewshots + [context]:
            interactions = render_fix_context(shot)
            for index, interaction in enumerate(interactions):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=interaction))
                else:
                    messages.append(AIMessage(content=interaction))
        debug_messages(*messages[-2:])

        response = self.llm(messages)
        debug_messages(response)
        explanation, observation, code = parse_fix(response.content)
        if "THE CODE IS CORRECT." in observation:
            print("The code is believed to be correct. No need to fix it.")
            return
        if not code.strip():
            print(
                "Generated code is empty. Please retry the fix with a more specific hint."
            )
            return

        post_context = copy.deepcopy(context)
        last_interaction = cast(InteractionIncomplete, post_context["interactions"][-1])
        post_context["interactions"][-1] = Interaction(
            **last_interaction,
            explanation=explanation,
            observation=observation,
            code=code,
        )
        return post_context

    def suggest(self, codes: list[str]) -> list[str]:
        human_message = render_ipython_cells(codes)
        messages = [
            SystemMessage(content=SUGGEST_INSTRUCTION),
            HumanMessage(content=human_message),
        ]
        debug_messages(*messages)
        response = self.llm(messages)
        suggestions = re.split(r"\d+\.\s+", response.content)
        suggestions = [s.strip().replace("\n", " ") for s in suggestions if s.strip()]
        debug_messages(response)

        return suggestions

    def explain(self, code: str) -> str:
        messages = [
            SystemMessage(content=EXPLAIN_INSTRUCTION),
            HumanMessage(content=code),
        ]
        debug_messages(*messages)
        response = self.llm(messages)
        debug_messages(response)
        return response.content

    def static_check(
        self, code: str, context: GenerateContext | FixContext
    ) -> tuple[bool | None, str]:
        # Check the quality of code by looking at it (i.e., rubberduck)
        messages = [
            SystemMessage(content=CHECK_INSTRUCTION),
            HumanMessage(content=render_check_context(code, context)),
        ]
        debug_messages(*messages)
        response = self.llm(messages)
        debug_messages(response)
        reason, last_line = response.content.rstrip().rsplit("\n", 1)
        if "INCORRECT" in last_line.upper():
            return False, reason
        if "CORRECT" in last_line.upper():
            return True, reason
        return None, response.content

    def output_sanity_check(
        self,
        code: str,
        context: GenerateContext | FixContext,
        error: str | None,
        output: str | None,
    ) -> tuple[bool | None, str]:
        # Run a sanity check of the output of the code
        messages = [
            SystemMessage(content=SANITY_CHECK_INSTRUCTION),
            HumanMessage(
                content=render_sanity_check_context(code, context, error, output)
            ),
        ]
        debug_messages(*messages)
        response = self.llm(messages)
        debug_messages(response)
        reason, last_line = response.content.rstrip().rsplit("\n", 1)
        if "INCORRECT" in last_line.upper():
            return False, reason
        if "CORRECT" in last_line.upper():
            return True, reason
        return None, response.content

    def visualization_check(
        self,
        request: str,
        previous_code: str,
        svg_string: str,
        variable_descriptions: dict[str, str],
        source,
    ) -> tuple[bool | None, list[tuple[bool | None, str]]]:
        vis_verifier = VisVerifier(self.llm, self)
        verifications = vis_verifier.verify(
            request, previous_code, svg_string, variable_descriptions, source
        )

        answers = [verification["answer"] for verification in verifications]
        if False in answers:
            pass_verify = False
        elif None in answers:
            pass_verify = None
        else:
            pass_verify = True

        reason = []
        for verification in verifications:
            answer = verification["answer"]
            aspect = verification["aspect"].capitalize()
            rationale = verification["rationale"]
            reason.append((answer, aspect + ": " + rationale))
        return pass_verify, reason
