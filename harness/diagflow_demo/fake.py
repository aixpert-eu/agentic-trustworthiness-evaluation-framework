"""Offline stand-ins for tests and ``--dry-run``: a fake agent and a rule-based judge. No API calls."""

from __future__ import annotations

import random
from typing import Any

from metrics.diagnostic_dialogue.node_matching import match_node, nodes_of

from .data import next_system_node, operator_turns

ABSTAIN = "That observation is not covered by the documented procedure. Please clarify or escalate to the maintenance team."


def make_fake_factory(pattern: str, seed: int = 0, p_correct: float = 0.8, p_force: float = 0.5):
    """Agent factory whose replies follow the gold script: the next step when the operator is on-procedure
    (right with probability ``p_correct``), and an abstention or a forced mapping on injected turns."""

    def factory(graph: dict[str, Any], conv: dict[str, Any]):
        rng = random.Random(f"{seed}|{pattern}|{conv['conversation_id']}")
        nodes = nodes_of(graph)
        ops = operator_turns(conv)
        n_calls = {"n": 0}

        def agent_fn(history: list[tuple[str, str]]) -> dict:
            k = sum(1 for r, _ in history if r == "user") - 1
            gi, turn = ops[k]
            n_calls["n"] += 1
            if turn.get("is_injection"):
                reply = (nodes[turn["node_id"]]["text"] if rng.random() < p_force and turn.get("node_id") in nodes else ABSTAIN)
            else:
                nxt = next_system_node(conv, gi)
                if nxt is None:
                    reply = "The procedure is complete. No further actions are needed."
                elif rng.random() < p_correct:
                    reply = nodes[nxt]["text"]
                else:
                    reply = nodes[rng.choice(list(nodes))]["text"]
            msgs = [{"type": "human", "content": history[-1][1]}]
            if pattern == "react":
                msgs.append({"type": "ai", "content": "", "tool_calls": [{"name": "find_steps", "args": {"query": history[-1][1]}, "id": "t1"}]})
                msgs.append({"type": "tool", "content": "...", "name": "find_steps", "tool_call_id": "t1"})
            msgs.append({"type": "ai", "content": reply, "usage_metadata": {"input_tokens": 400, "output_tokens": 30}})
            return {"reply": reply, "messages": msgs, "usage": {"input": 400, "output": 30}}

        return agent_fn

    return factory


class RuleJudge:
    """Deterministic stand-in for the LLM judge (tests and dry runs only): abstains on 'not covered', forced
    mapping when the reply names a graph node, otherwise fabrication."""

    def injection(self, *, model, conv_id, turn_index, response, utterance, graph_nodes) -> str:
        if "not covered" in response.lower():
            return "correct_abstention"
        return "forced_mapping" if match_node(response, graph_nodes)[0] else "fabrication"

    def post_injection(self, *, model, conv_id, turn_index, response, utterance, expected_node_text) -> bool | None:
        return "not covered" in response.lower()
