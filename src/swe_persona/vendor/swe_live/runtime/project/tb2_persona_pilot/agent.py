#!/usr/bin/env python3
"""The system under test: GPT6 as a terminal coding agent inside the task container.

Deliberately minimal so the episode measures the benchmark, not a harness: one
shell tool, one user-facing reply, an explicit STATUS field so
`completion_claim` is a typed event rather than a regex guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dockerx import Container
from llm import Llm

SYSTEM_TEMPLATE = """You are a coding agent working in a Linux container through a shell. A real user is on the other side of the conversation: they opened this task, they will answer questions between your turns, and their acceptance decides whether the work is done.

## How to act
Emit exactly ONE action per step, in one of these two forms, and nothing else:

THOUGHT: <one or two lines>
ACTION: shell
```sh
<commands to run in bash; keep each step small and check the output>
```

or

THOUGHT: <one or two lines>
ACTION: reply_to_user
STATUS: working
MESSAGE:
<what you tell the user; this ends your turn and the user replies next>

Use `STATUS: done` only when you believe every requirement the user has stated is implemented AND verified by you. Use `STATUS: working` when you are answering, asking, or reporting progress.

## Rules
- You have at most {max_steps} shell steps per turn; the turn ends when you reply to the user.
- Commands run as root with `bash -lc` in {workdir}. There is no interactive input.
- The user cannot see your shell output. If a number matters to them, put it in the reply.
- Do the work in the container; a plan alone is never a deliverable.

## Environment (facts, not suggestions)
{environment}
"""

STEP_NUDGE = (
    "You have used your shell budget for this turn. Emit a `reply_to_user` action now, with the state of "
    "the work and anything you need from the user."
)
FORMAT_NUDGE = (
    "Your last output did not parse. Emit exactly one action: either `ACTION: shell` with one ```sh``` block, "
    "or `ACTION: reply_to_user` with `STATUS:` and `MESSAGE:`."
)


@dataclass
class Step:
    index: int
    command: str
    exit_code: int
    stdout: str
    thought: str = ""


@dataclass
class AgentTurn:
    turn: int
    message_id: str
    reply: str
    status: str
    steps: List[Step] = field(default_factory=list)
    raw_outputs: List[str] = field(default_factory=list)
    parse_failures: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "message_id": self.message_id,
            "reply": self.reply,
            "status": self.status,
            "parse_failures": self.parse_failures,
            "steps": [
                {"index": s.index, "thought": s.thought, "command": s.command,
                 "exit_code": s.exit_code, "stdout": s.stdout}
                for s in self.steps
            ],
        }


class TerminalAgent:
    def __init__(
        self,
        llm: Llm,
        container: Container,
        environment_card: str,
        max_steps_per_turn: int = 14,
        max_output_chars: int = 3000,
        workdir: str = "/app",
        command_timeout: float = 900.0,
    ) -> None:
        self.llm = llm
        self.container = container
        self.max_steps = max_steps_per_turn
        self.max_output_chars = max_output_chars
        self.workdir = workdir
        self.command_timeout = command_timeout
        self.system_prompt = SYSTEM_TEMPLATE.format(
            max_steps=max_steps_per_turn, workdir=workdir, environment=environment_card
        )
        self.history: List[Dict[str, str]] = []

    # ---- protocol -------------------------------------------------------
    @staticmethod
    def _parse(text: str) -> Tuple[str, Dict[str, str]]:
        thought = ""
        m = re.search(r"THOUGHT:\s*(.+?)(?:\n(?:ACTION|MESSAGE|STATUS):|\Z)", text, re.DOTALL)
        if m:
            thought = m.group(1).strip()
        action = re.search(r"ACTION:\s*([a-zA-Z_]+)", text)
        kind = (action.group(1).strip().lower() if action else "")
        if kind == "shell":
            block = re.search(r"```(?:sh|bash|shell)?\s*\n(.*?)```", text, re.DOTALL)
            if not block:
                return "invalid", {"thought": thought}
            return "shell", {"thought": thought, "command": block.group(1).strip()}
        if kind in ("reply_to_user", "reply"):
            status = re.search(r"STATUS:\s*([a-zA-Z_]+)", text)
            msg = re.search(r"MESSAGE:\s*\n?(.*)\Z", text, re.DOTALL)
            body = (msg.group(1).strip() if msg else text.strip())
            return "reply", {
                "thought": thought,
                "status": (status.group(1).strip().lower() if status else "working"),
                "message": body,
            }
        # No ACTION line but a block that looks like commands -> treat as shell
        block = re.search(r"```(?:sh|bash|shell)\s*\n(.*?)```", text, re.DOTALL)
        if block:
            return "shell", {"thought": thought, "command": block.group(1).strip()}
        return "invalid", {"thought": thought}

    def _trim(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        return text[:half] + f"\n...[{len(text) - self.max_output_chars} chars trimmed]...\n" + text[-half:]

    # ---- one user turn --------------------------------------------------
    def run_turn(self, user_message: str, turn: int, step_budget: Optional[int] = None) -> AgentTurn:
        budget = step_budget or self.max_steps
        self.history.append({"role": "user", "content": user_message})
        result = AgentTurn(turn=turn, message_id=f"a-{turn}", reply="", status="working")
        steps_used = 0
        format_retries = 0
        while True:
            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            text = self.llm.text(messages, role=f"agent.turn{turn}", max_output_tokens=None)
            result.raw_outputs.append(text)
            self.history.append({"role": "assistant", "content": text})
            kind, payload = self._parse(text)

            if kind == "reply":
                result.reply = payload["message"]
                result.status = "done" if payload["status"].startswith("done") else "working"
                return result

            if kind == "shell":
                if steps_used >= budget:
                    self.history.append({"role": "user", "content": STEP_NUDGE})
                    continue
                steps_used += 1
                rc, out, err = self.container.sh(payload["command"], timeout=self.command_timeout, workdir=self.workdir)
                combined = self._trim(((out or "") + (("\n[stderr]\n" + err) if err.strip() else "")).strip())
                step = Step(
                    index=steps_used, command=payload["command"], exit_code=rc,
                    stdout=combined, thought=payload.get("thought", ""),
                )
                result.steps.append(step)
                feedback = f"[exit {rc}]\n{combined if combined else '(no output)'}"
                if steps_used >= budget:
                    feedback += "\n\n" + STEP_NUDGE
                self.history.append({"role": "user", "content": feedback})
                continue

            result.parse_failures += 1
            format_retries += 1
            if format_retries > 3:
                result.reply = text.strip()[:2000]
                result.status = "working"
                return result
            self.history.append({"role": "user", "content": FORMAT_NUDGE})
