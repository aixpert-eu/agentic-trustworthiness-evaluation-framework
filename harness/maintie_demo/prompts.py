"""Prompts for the two agent configurations. Wording is deliberately plain;
both agents get the same label definitions so the comparison is about the
pattern and the model, not about prompt quality."""

from __future__ import annotations

LABEL_GUIDE = """\
Entity classes (use exactly these names):
- PhysicalObject: equipment, parts, substances, people (pump, hydraulic hose, seal, oil, fitter).
- State: a condition of an object (broken, worn, not working, seized, faulty).
- Process: something an object does or undergoes (leaking, overheating, vibrating, tripping).
- Activity: maintenance or supporting work done by people (replace, inspect, repair, clean, isolate).
- Property: an attribute of an object (crack, noise, hole, pressure).

Relation types (use exactly these names; head -> tail):
- hasPart: whole -> part (air conditioner -> thermostat).
- contains: container -> contents (tank -> oil).
- hasProperty: object -> property (hose -> crack).
- isA: specific -> general (hydraulic hose -> hose).
- hasParticipant/hasPatient: State, Process or Activity -> the object it affects (replace -> seal; broken -> pump).
- hasParticipant/hasAgent: State, Process or Activity -> the object that performs or causes it (leaking -> pump when the pump is the source).

Entity "text" must be copied from the work order exactly as written. Annotate every entity mention, including states, processes and activities, not only equipment. Do not invent entities or relations that the text does not support."""

REACT_ROLE = "Maintenance work-order annotator"

REACT_TASK = (
    "Annotate one short maintenance work order. You receive its text as `text`. "
    "Use the ontology_lookup tool when you are unsure which class a term belongs to. "
    "When finished, call finish_task with the entities and relations.\n\n" + LABEL_GUIDE
)

RAG_QUERY_SYSTEM = (
    "You annotate maintenance work orders. Call the retrieval tool once with a short query "
    "to find similar, already annotated work orders. Do not answer yet."
)

RAG_ANSWER_SYSTEM = (
    "You annotate maintenance work orders. Use the retrieved annotated examples as guidance, "
    "then annotate the work order in the conversation. Do NOT call any tool. "
    "Reply with a single JSON object and nothing else, in this form:\n"
    '{"entities": [{"text": "...", "type": "..."}], '
    '"relations": [{"head": "...", "tail": "...", "type": "..."}]}\n\n' + LABEL_GUIDE
)


def rag_prompt_overrides() -> dict:
    """Value for BasicRAG's ``prompt_overrides`` kwarg (chat-message lists)."""
    return {
        "query_generation_prompt": [("system", RAG_QUERY_SYSTEM)],
        "answer_generation_prompt": [("system", RAG_ANSWER_SYSTEM)],
    }
