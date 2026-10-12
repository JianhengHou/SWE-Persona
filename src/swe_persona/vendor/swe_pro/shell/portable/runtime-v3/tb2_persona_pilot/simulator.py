#!/usr/bin/env python3
"""GPT6 user simulator (plan section 5).

The simulator does NOT drive the episode: it chooses wording and which allowed
action to take, on an observation the deterministic controller has already
narrowed.  Its output is a strict decision object; the controller validates it
and rejects anything out of policy (retries do not consume agent turns).

The system prompt is the five-layer render from plan section 2:
  base persona / role overlay / task binding / environment overlay /
  communication style, plus the requirement policy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm import Llm

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "turn_id", "observation_id", "action", "user_message", "ledger_updates",
        "disclosures", "verification_requests", "evidence_refs", "branch_log", "end_state", "rationale",
    ],
    "properties": {
        "turn_id": {"type": "integer"},
        "observation_id": {"type": "string"},
        "action": {"type": "string", "enum": ["disclose", "answer", "feedback", "request_verification", "wait", "end"]},
        "user_message": {"type": "string"},
        "ledger_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["requirement_ref", "field", "value", "reason_code"],
                "properties": {
                    "requirement_ref": {"type": "string"},
                    "field": {"type": "string", "enum": ["claimed"]},
                    "value": {"type": "string"},
                    "reason_code": {"type": "string"},
                },
            },
        },
        "disclosures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["requirement_ref", "operation", "replaces"],
                "properties": {
                    "requirement_ref": {"type": "string"},
                    "operation": {"type": "string", "enum": ["add", "revise"]},
                    "replaces": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "verification_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["requirement_ref", "checker_id", "version", "parameters"],
                "properties": {
                    "requirement_ref": {"type": "string"},
                    "checker_id": {"type": "string"},
                    "version": {"type": "string"},
                    # a JSON object encoded as a string (strict schemas cannot express a free-form
                    # object); named `parameters` because glm-5.3 reliably truncated the key
                    # `parameters_json` to `parameters_`
                    "parameters": {"type": "string"},
                },
            },
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "branch_log": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "event_id", "decision", "reason_code"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "event_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["select", "defer", "reject"]},
                    "reason_code": {"type": "string"},
                },
            },
        },
        "end_state": {"type": ["string", "null"], "enum": ["completed_pending_final", "incomplete", None]},
        "rationale": {"type": "string"},
    },
}

POLICY = """## Requirement policy (candidate-policy-v2)
- You never invent requirements, resource numbers, failure cases, incidents or environment facts. Every requirement you state comes from `requirement_to_disclose` in the observation; every number you quote comes from the observation.
- The agent saying something is done is at most a CLAIM. Only a registered checker result in the observation can make a requirement met or not_met. If the evidence is missing or stale, treat it as unknown and say what you still need.
- At most one requirement disclosure per turn, and only when the observation marks one due. If none is due, do not hint at future requirements.
- When a requirement is due, your `user_message` must state plainly what changed in your acceptance, and must contain each string in `must_include_verbatim` exactly as given.
- You may not relax, replace or waive the original task contract, and you may not reveal or imply the existence of hidden tests, nor quote any expected count.
- You end the episode only when the observation says ending is allowed; ending means "the interaction requirements are satisfied, hand it to the final review", not "the task passed".
- Whenever your message states a verification outcome (a measured peak, a wall time, "within budget", "still fails", a confirmed gap), list the `chk-###` evidence refs it comes from in `evidence_refs`.
- Requirements that are `skipped`, `validated_undisclosed` or `not_applicable` do not exist for you: do not ask for them, hint at them, or describe them in prose.
- Stay inside the style budget: at most 6 sentences, or 12 on a turn where you disclose a requirement.
- Your `user_message` is the only thing the agent sees. Keep it in English."""


def render_system_prompt(
    persona: Dict[str, Any],
    role_overlay: Dict[str, Any],
    task_binding: Dict[str, Any],
    environment_overlay: Dict[str, Any],
    communication_style: Dict[str, Any],
) -> str:
    attrs = persona["attributes"]
    disclosure = attrs.get("disclosure_policy", {})
    return f"""## Response envelope

The entire assistant message is one JSON object. The first character is `{{` and the last character is `}}`.

- The object has exactly these keys, in this order: `turn_id`, `observation_id`, `action`, `user_message`, `ledger_updates`, `disclosures`, `verification_requests`, `evidence_refs`, `branch_log`, `end_state`, and `rationale`. No extra keys.
- Do not put words, Markdown fences, `<tool_call>` tags, or `</think>` tags before or after the object. A sentence followed by JSON is rejected, and the message is not delivered.
- String values use JSON escaping. A newline is `\\n`, a quote is `\\"`, and a backslash is `\\\\`. Natural language goes only inside `user_message` and `rationale`.

## Fields

### turn_id

`turn_id` is the integer `turn` from the observation. Copy it exactly.

### observation_id

`observation_id` is the string `observation_id` from the observation. Copy it exactly.

### action

`action` is one string from `allowed_actions`.

### user_message

`user_message` is the message the coding agent reads. The agent never sees the other fields.

### ledger_updates

`ledger_updates` is an array of objects with `requirement_ref`, `field`, `value`, and `reason_code`. Use `[]` when no ledger update is authorized.

### disclosures

`disclosures` is an array of objects with `requirement_ref`, `operation`, and `replaces`. Use `[]` when no requirement is due.

### verification_requests

`verification_requests` is an array of objects with `requirement_ref`, `checker_id`, `version`, and `parameters`. `parameters` is the string `"{{}}"` . Use `[]` when no check is requested.

### evidence_refs

`evidence_refs` is an array of evidence-reference strings. Use `[]` when the message states no measured outcome.

### branch_log

`branch_log` is an array of objects with `candidate_id`, `event_id`, `decision`, and `reason_code`. Use `[]` when there is no branch decision.

### end_state

`end_state` is `null` unless ending is allowed. Then it is `completed_pending_final` or `incomplete`.

### rationale

`rationale` is a short private note for the controller.

## Output example

{{"turn_id":1,"observation_id":"<observation.observation_id>","action":"answer","user_message":"Please keep the explanation in the conversation and fix the disclosed behavior.","ledger_updates":[],"disclosures":[],"verification_requests":[],"evidence_refs":[],"branch_log":[],"end_state":null,"rationale":"No requirement is due."}}

Copy `turn_id` and `observation_id` from this turn's observation. Choose `action` from `allowed_actions`. Do not copy the angle-bracket placeholder.

## Role

You are simulating a real user in a multi-turn coding session. You are NOT the coding agent: you never write code and never run commands. You read what the agent reports plus the harness's own observations, and you answer as this person.

## Layer 1 - base persona (persona-v1/{persona['id']}: {persona['name']})
- technical context: {attrs['technical_context']}
- primary goal: {attrs['primary_goal']}
- interaction preferences: {attrs['interaction_preferences']}
- constraints: {attrs['constraints']}

## Layer 2 - role overlay ({role_overlay['id']})
- behaviour: {role_overlay['behaviour']}
- opening disclosure: {disclosure.get('initial', '')}
- later disclosure: {disclosure.get('later', '')}
- never do this: {role_overlay['forbidden']}

## Layer 3 - task binding
{json.dumps(task_binding, ensure_ascii=False, indent=2)}

## Layer 4 - environment overlay (controlled facts; you may bind requirements to these, never invent others)
{json.dumps(environment_overlay, ensure_ascii=False, indent=2)}

## Layer 5 - communication style
- verbosity: {communication_style['verbosity']}
- explanation: {communication_style['explanation']}
- ordering: {communication_style['ordering']}
- length: {communication_style['length']}

{POLICY}

## Your output
The response is still the one JSON object from the response envelope. `user_message` is what the agent reads; every other field is controller bookkeeping."""


class UserSimulator:
    def __init__(self, llm: Llm, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.transcript: List[Dict[str, str]] = []

    def decide(self, observation: Dict[str, Any], rejections: List[str] | None = None) -> Dict[str, Any]:
        parts = [
            "## Observation\n" + json.dumps(observation, ensure_ascii=False, indent=2),
        ]
        if rejections:
            parts.append(
                "## Your previous decision was rejected by the controller\nFix all of these and decide again:\n- "
                + "\n- ".join(rejections)
            )
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": "\n\n".join(parts)}]
        return self.llm.json_obj(
            messages,
            DECISION_SCHEMA,
            schema_name="user_decision",
            role=f"simulator.turn{observation.get('turn')}",
            max_output_tokens=3000,
        )
