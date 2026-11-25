"""
Builds the two real WP4 patterns (ReAct, Basic RAG) for the DiagFlowBench demo, on one procedure graph.

Needs WP4's `inobo` package (currently private; install it with ``uv sync --extra inobo``). capture.py imports
this file lazily and only for the react and basic_rag patterns without ``--dry-run``, so the dry run and the
public ``baseline`` pattern work without `inobo`.

Requires OPENROUTER_API_KEY in the environment (never put the key in a file).
Uses the same WP4 API as the MaintIE demo's wp4_agents.py.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from inobo.patterns.agent_orchestration.react_agent.pattern import ReactAgent
from inobo.patterns.knowledge_retrieval.basic_rag.pattern import BasicRAG
from inobo.utils.schemas import BasicRagState, LLMConfig
from pydantic import BaseModel, Field

from harness.adapters import _last_ai_text

from . import prompts
from .retrieval import GraphIndex


class DialogueInput(BaseModel):
    history: str = Field(description="The conversation so far, one line per turn.")
    operator_message: str = Field(description="The operator's latest message.")


class Reply(BaseModel):
    reply: str = Field(description="The assistant's reply: one next diagnostic action in 1 to 3 sentences, or an abstention.")


class RetrievalInput(BaseModel):
    query: str = Field(description="Short search query for the relevant procedure steps.")


def _llm(model: str, temperature: float) -> LLMConfig:
    return LLMConfig(model=model, model_provider="openrouter", config={"temperature": temperature})


def _field(obj, name: str) -> str:
    return getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)


def build_agent(pattern: str, model: str, graph: dict, *, temperature: float = 0.0):
    """Return ``agent_fn(history) -> {"reply", "messages"}`` for 'react' or 'basic_rag' on one procedure graph."""
    index = GraphIndex(graph)

    if pattern == "react":
        def find_steps(query: str) -> str:
            """Search the documented procedure for steps matching a description. Returns step ids with their text."""
            return index.find_steps(query)

        def get_step(step_id: str) -> str:
            """Get one step of the procedure by its id, with the steps that follow it."""
            return index.describe(step_id)

        def next_steps(step_id: str) -> str:
            """List what follows a step, with the Yes or No label of each branch."""
            return index.next_steps(step_id)

        agent = ReactAgent(
            agent_role=prompts.REACT_ROLE, task_description=prompts.REACT_TASK,
            input_schema=DialogueInput, output_schema=Reply,
            reasoning_llm_config=_llm(model, temperature), tools=[find_steps, get_step, next_steps],
        )

        def agent_fn(history):
            past = "\n".join(f"{'Operator' if r == 'user' else 'Assistant'}: {c}" for r, c in history[:-1])
            res = agent.run({"history": past or "(start of conversation)", "operator_message": history[-1][1]})
            return {"reply": _field(res["structured_response"], "reply") or "", "messages": res["messages"]}

        return agent_fn

    if pattern == "basic_rag":
        def retrieve_steps(query: str) -> str:
            """Retrieve procedure steps relevant to the query, with the steps that follow them."""
            return index.retrieve(query, k=4)

        def make_rag(overrides):
            return BasicRAG(
                retriever=retrieve_steps, retriever_input_schema=RetrievalInput, retriever_tool_name="retrieve_steps",
                retriever_tool_description="Retrieve steps of the documented troubleshooting procedure.",
                query_rewriter_llm_config=_llm(model, temperature), answer_generator_llm_config=_llm(model, temperature),
                prompt_overrides=overrides,
            )

        def usable(r):  # the pattern calls .format_messages() on the stored prompt; check it without any API call
            try:
                for name in ("query_generation_prompt", "answer_generation_prompt"):
                    r.get_prompt(name).format_messages()
                return True
            except AttributeError:
                return False

        texts = {"query_generation_prompt": prompts.RAG_QUERY_SYSTEM, "answer_generation_prompt": prompts.RAG_ANSWER_SYSTEM}
        # WP4 versions differ in what prompt_overrides must be: a message list (documented) or a ChatPromptTemplate.
        rag = make_rag({k: [SystemMessage(content=v)] for k, v in texts.items()})
        if not usable(rag):
            rag = make_rag({k: ChatPromptTemplate.from_messages([SystemMessage(content=v)]) for k, v in texts.items()})
        if not usable(rag):
            raise RuntimeError("BasicRAG rejected both prompt_overrides formats; check the installed WP4 version.")

        def agent_fn(history):
            msgs = [HumanMessage(content=c) if r == "user" else AIMessage(content=c) for r, c in history]
            res = rag.run(BasicRagState(messages=msgs, context=[]))
            return {"reply": _last_ai_text(res["messages"]) or "", "messages": res["messages"]}

        return agent_fn

    raise ValueError(f"unknown pattern {pattern!r}")
