"""User simulator (TB3 v6 section 5).

It sees the persona, the public task, the conversation, which original facts and
requirements are due, and for each active requirement only the judge's verdict and
user-facing feedback.  It never sees code, tool output, checker results, hidden tests,
the reference solution or requirements that are not due.  The controller validates
every decision and substitutes a deterministic message after repeated failures.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import Llm, LlmSchemaError

ACTIONS = ("answer", "disclose_requirement", "request_fix", "request_evidence", "done")


def decision_schema(allowed: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "user_message", "satisfied", "rationale"],
        "properties": {
            "action": {"type": "string", "enum": list(allowed or ACTIONS)},
            "user_message": {"type": "string"},
            "satisfied": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    }


DECISION_PROTOCOL = """## Response envelope

The entire assistant message is one JSON object. The first character is `{` and the last character is `}`.

- The object has exactly these keys, in this order: `action`, `user_message`, `satisfied`, and `rationale`. No extra keys.
- Do not put words, Markdown fences, `<tool_call>` tags, or `</think>` tags before or after the object. A sentence followed by JSON is rejected, and the message is not delivered.
- `user_message` and `rationale` are JSON strings. A newline is `\\n`, a quote is `\\"`, and a backslash is `\\\\`.

## Fields

### action

`action` is one string from `allowed_actions` in the observation.

### user_message

`user_message` is the natural-language message the coding agent reads. The agent never sees text outside this string.

### satisfied

`satisfied` is a JSON boolean. Use `true` only when `action` is `done` and the observation allows done. Otherwise use `false`.

### rationale

`rationale` is a short private note. The coding agent never sees it.

## Output example

{"action":"answer","user_message":"Please continue with the approach you described.","satisfied":false,"rationale":"No requirement is due."}

Replace `action` with one action allowed on this turn. Keep the four keys.
"""


def render_messages(prompt: str, persona_card: Dict[str, Any], observation: Dict[str, Any],
                    rejected: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    system = DECISION_PROTOCOL + "\n" + prompt + "\n\n## Your persona\n" + json.dumps(persona_card, ensure_ascii=False, indent=1)
    user = ("Decide your next message to the coding agent.  Reply with one JSON object.\n\n"
            + json.dumps(observation, ensure_ascii=False, indent=1))
    if rejected is not None:
        user += ("\n\n## Your previous decision was rejected\n"
                 + json.dumps(rejected["decision"], ensure_ascii=False)[:4000]
                 + "\n\nFix every problem:\n- " + "\n- ".join(rejected["errors"]))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class UserSimulator:
    def __init__(self, llm: Llm, prompt: str, persona_card: Dict[str, Any]) -> None:
        self.llm = llm
        self.prompt = prompt
        self.persona_card = persona_card

    def decide(self, observation: Dict[str, Any], validate, attempts: int = 4) -> Dict[str, Any]:
        """Return {"decision", "attempts", "fallback"}; `validate` returns a list of errors."""
        schema = decision_schema(observation["allowed_actions"])
        rejected = None
        log: List[Dict[str, Any]] = []
        for attempt in range(attempts):
            messages = render_messages(self.prompt, self.persona_card, observation, rejected)
            try:
                decision = self.llm.json(messages, schema, "user_decision")
            except LlmSchemaError as exc:
                log.append({"attempt": attempt, "errors": exc.errors})
                rejected = {"decision": exc.text[:2000], "errors": exc.errors}
                continue
            errors = validate(decision, observation)
            log.append({"attempt": attempt, "errors": errors, "decision": decision})
            if not errors:
                return {"decision": decision, "attempts": log, "fallback": False}
            rejected = {"decision": decision, "errors": errors}
        return {"decision": None, "attempts": log, "fallback": True}
