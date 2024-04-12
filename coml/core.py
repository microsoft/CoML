from __future__ import annotations

import copy
import random
import re
import warnings
from typing import Any, Callable, Literal, TypeVar, cast

import colorama
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from scipy.spatial.distance import cosine as cosine_distance

from .prompt_utils import (
    CHECK_INSTRUCTION,
    EXPLAIN_INSTRUCTION,
    FIX_INSTRUCTION,
    GENERATE_INSTRUCTION,
    GENERATE_INSTRUCTION_COT,
    GENERATE_INSTRUCTION_VIS_MATPLOTLIB,
    GENERATE_INSTRUCTION_VIS_SEABORN,
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

_Type = TypeVar("_Type")


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
    """
    CoML agent that accepts data science requests and generates code.

    Attributes:
        llm: The language model that generates responses.
        prompt_version: The version of prompt to use (can be ``v1`` or ``v2``).
        prompt_validation: A function that takes a list of messages and returns
            whether the prompt is valid, which is useful for limiting the number of
            tokens in the prompt.
        num_examples: The number of examples to show in the prompt. It can be a
            number between 0 and 1, interpreted as the percentage of examples to show.
            It can also be an integer, interpreted as the number of examples to show.
        message_style: Can be ``chatgpt`` in which system messages are shown, or
            ``gemini`` in which only human and ai messages are shown.
        chain_of_thought: Whether to use chain of thought (COT) in the prompt.
        context_order: The order of the context in the prompt. Default to ``vcr``.
            ``v`` for variable descriptions, ``c`` for codes, ``r`` for request.
        ensemble: Perform ``ensemble`` number of LLM calls and ensemble the results.
        ensemble_shuffle: Shuffle the examples in the prompt before ensemble.
        example_ranking: A model that ranks the examples. If provided, the examples
            will be ranked by the model before selecting the examples.
        intact_instruction: Whether to instruct LLM to keep the variables unmodified.
            For experimenting purposes only.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_version: Literal[
            "v1", "v2", "kaggle", "leetcode", "matplotlib", "seaborn"
        ] = "v2",
        prompt_validation: Callable[[list[BaseMessage]], bool] | None = None,
        num_examples: float | int = 1.0,
        message_style: Literal["chatgpt", "gemini"] = "chatgpt",
        chain_of_thought: bool = False,
        context_order: Literal[
            "vcr", "cvr", "rvc", "rcv", "vr", "rv", "cr", "rc", "r"
        ] = "vcr",
        ensemble: int | None = None,
        ensemble_shuffle: bool = True,
        example_ranking: Embeddings | None = None,
        intact_instruction: bool = True,
    ):
        self.llm = llm
        self.prompt_version = prompt_version
        self.prompt_validation = prompt_validation
        self.num_examples = num_examples
        self.message_style = message_style
        self.chain_of_thought = chain_of_thought
        self.context_order = context_order
        self.ensemble = ensemble
        self.ensemble_shuffle = ensemble_shuffle
        self.example_ranking = example_ranking
        self.intact_instruction = intact_instruction

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

    def _pre_generation(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if self.message_style == "gemini":
            # Merge the first two messages.
            if len(messages) > 1 and isinstance(messages[1], HumanMessage):
                messages1_content = cast(str, messages[1].content)
                if not messages1_content.startswith("### Task Start ###"):
                    messages1_content = "### Task Start ###\n\n" + messages1_content
                messages[1] = HumanMessage(
                    content=cast(str, messages[0].content) + "\n\n" + messages1_content
                )
                messages = messages[1:]
            else:
                messages[0] = HumanMessage(content=cast(str, messages[0].content))

        if self.prompt_validation is not None and not self.prompt_validation(messages):
            raise ValueError("Prompt validation failed.")

        return messages

    def _ensemble_generate(self, messages: list[BaseMessage]) -> BaseMessage:
        """Ensemble the result from multiple LLM calls."""

        if not self.ensemble:
            return self._generate(messages)

        results: list[tuple[float, BaseMessage]] = []
        for _ in range(self.ensemble):
            if self.ensemble_shuffle and len(messages) > 2:
                # Shuffle the examples
                first_message = messages[0]
                interactions = messages[1:]

                start_indices = []
                # Can be [Task A Human, AI, Human AI, Task B Human, AI]
                for index, message in enumerate(interactions):
                    if isinstance(message, HumanMessage) and cast(
                        str, message.content
                    ).startswith("### Task Start ###"):
                        start_indices.append(index)

                # Can be [Human, AI, Human AI]
                if not start_indices:
                    # Loosen the constraint and find all human messages
                    start_indices = [
                        index
                        for index, message in enumerate(interactions)
                        if isinstance(message, HumanMessage)
                    ]

                groups = [
                    interactions[index:index_next]
                    for index, index_next in zip(start_indices, start_indices[1:])
                ]

                # Shuffle the groups and combine them
                random.shuffle(groups)
                messages = (
                    [first_message]
                    + [message for group in groups for message in group]
                    + interactions[start_indices[-1] :]
                )

            messages = self._pre_generation(messages)
            result = self.llm.generate([messages], logprobs=True)
            message = result.generations[0][0].message  # type: ignore
            generation_info = result.generations[0][0].generation_info
            if generation_info is None or "logprobs" not in generation_info:
                raise ValueError("Logprobs not found in generation_info.")
            logprobs = [
                content["logprob"] for content in generation_info["logprobs"]["content"]
            ]
            if not logprobs:
                mean_logprobs = float("-inf")
            else:
                mean_logprobs = sum(logprobs) / len(logprobs)

            results.append((mean_logprobs, message))

        results.sort(key=lambda x: x[0], reverse=True)

        return results[0][1]

    def _generate(self, messages: list[BaseMessage]) -> BaseMessage:
        """Generate a response from the LLM."""
        messages = self._pre_generation(messages)
        return self.llm(messages)

    def _select_examples(self, query: str, fewshots: list[_Type]) -> list[_Type]:
        """Select examples from the fewshots."""
        if self.num_examples == 0:
            return []

        if self.example_ranking is not None:
            documents = [cast(Any, shot).get("request", "N/A") for shot in fewshots]
            embeddings = self.example_ranking.embed_documents(documents)
            # Use embed_documents instead of embed_query because the latter has cache
            query_embedding = self.example_ranking.embed_documents([query])[0]
            similarities = [
                (cosine_distance(query_embedding, embedding), shot)
                for embedding, shot in zip(embeddings, fewshots)
            ]
            similarities.sort(key=lambda x: x[0])
            fewshots = [shot for _, shot in similarities]

        if isinstance(self.num_examples, int):
            return fewshots[: self.num_examples]
        else:
            num_shots = max(int(len(fewshots) * self.num_examples), 1)
            return fewshots[:num_shots]

    def generate_code(
        self,
        request: str,
        variable_descriptions: dict[str, str],
        codes: list[str],
    ) -> GenerateContext:
        fewshots = cached_generate_fewshots(self.prompt_version)
        messages: list[BaseMessage] = []

        if self.chain_of_thought:
            generate_instruction = GENERATE_INSTRUCTION_COT
        else:
            generate_instruction = GENERATE_INSTRUCTION
        if not self.intact_instruction:
            generate_instruction = re.sub(
                r"- Do not overwrite or modify.*\n", "", generate_instruction
            )
            for shot in fewshots:
                if "answer_wo_intact" in shot:
                    shot["answer"] = shot.pop("answer_wo_intact")
                if "rationale_wo_intact" in shot:
                    shot["rationale"] = shot.pop("rationale_wo_intact")

        if self.prompt_version == "matplotlib":
            generate_instruction = GENERATE_INSTRUCTION_VIS_MATPLOTLIB
        elif self.prompt_version == "seaborn":
            generate_instruction = GENERATE_INSTRUCTION_VIS_SEABORN

        messages.append(SystemMessage(content=generate_instruction))

        for shot in self._select_examples(request, fewshots):
            question, answer = render_generate_context(
                shot, cot=self.chain_of_thought, context_order=self.context_order
            )
            messages.append(HumanMessage(content=question))
            if answer is not None:
                messages.append(AIMessage(content=answer))
        context = GenerateContextIncomplete(
            variables=variable_descriptions, codes=codes, request=request
        )
        question, _ = render_generate_context(
            context, cot=self.chain_of_thought, context_order=self.context_order
        )
        messages.append(HumanMessage(content=question))

        debug_messages(*messages)

        response = self._ensemble_generate(messages)
        debug_messages(response)

        if not isinstance(response.content, str):
            raise ValueError(f"Response is not a string: {response.content}")
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
        fewshots = self._select_examples(prev_context["request"] or "N/A", fewshots)
        messages: list[BaseMessage] = [
            SystemMessage(content=FIX_INSTRUCTION),
        ]
        context = self._fix_context_from_any_context(
            prev_context, error=error, output=output, hint=hint
        )
        for shot in fewshots + [context]:
            interactions = render_fix_context(shot, context_order=self.context_order)
            for index, interaction in enumerate(interactions):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=interaction))
                else:
                    messages.append(AIMessage(content=interaction))

        debug_messages(*messages)

        response = self._ensemble_generate(messages)
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
        response = self._generate(messages)
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
        response = self._generate(messages)
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
        response = self._generate(messages)
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
        response = self._generate(messages)
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
