"""
Builds the two real WP4 patterns (ReAct, Basic RAG) for the MaintIE demo.

Needs WP4's `inobo` package (currently private; install it with
``uv sync --extra inobo``). capture.py imports this file lazily and only
without ``--dry-run``, so the rest of the demo works without `inobo`.

Requires OPENROUTER_API_KEY in the environment (never put the key in a file).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from inobo.patterns.agent_orchestration.react_agent.pattern import ReactAgent
from inobo.patterns.knowledge_retrieval.basic_rag.pattern import BasicRAG
from inobo.utils.schemas import BasicRagState, LLMConfig
from pydantic import BaseModel, Field

from . import prompts
from .retrieval import BM25Index, OntologyLookup
from .schemas import Extraction


class WorkOrderInput(BaseModel):
    text: str = Field(description="The maintenance work-order text to annotate.")


class RetrievalInput(BaseModel):
    query: str = Field(description="Short search query describing the work order.")


def _llm(model: str, temperature: float) -> LLMConfig:
    return LLMConfig(model=model, model_provider="openrouter", config={"temperature": temperature})


def build_agent(pattern: str, model: str, *, silver, exclude_texts, temperature: float = 0.0):
    """Return ``agent_fn(text) -> WP4 result dict`` for 'react' or 'basic_rag'."""
    if pattern == "react":
        lookup = OntologyLookup(silver, exclude_texts)

        def ontology_lookup(term: str) -> str:
            """Return the most likely top-level ontology class(es) for a maintenance term, or 'unknown'."""
            return lookup.lookup(term)

        agent = ReactAgent(
            agent_role=prompts.REACT_ROLE,
            task_description=prompts.REACT_TASK,
            input_schema=WorkOrderInput,
            output_schema=Extraction,
            reasoning_llm_config=_llm(model, temperature),
            tools=[ontology_lookup],
        )
        return lambda text: agent.run({"text": text})

    if pattern == "basic_rag":
        index = BM25Index(silver, exclude_texts)

        def retrieve_examples(query: str) -> str:
            """Retrieve similar, already annotated work orders."""
            return index.retrieve(query, k=3)

        overrides = {
            name: [SystemMessage(content=text) for _, text in msgs]
            for name, msgs in prompts.rag_prompt_overrides().items()
        }
        rag = BasicRAG(
            retriever=retrieve_examples,
            retriever_input_schema=RetrievalInput,
            retriever_tool_name="retrieve_examples",
            retriever_tool_description="Retrieve similar, already annotated maintenance work orders.",
            query_rewriter_llm_config=_llm(model, temperature),
            answer_generator_llm_config=_llm(model, temperature),
            prompt_overrides=overrides,
        )
        return lambda text: rag.run(BasicRagState(messages=[HumanMessage(content=text)], context=[]))

    raise ValueError(f"unknown pattern {pattern!r}")
