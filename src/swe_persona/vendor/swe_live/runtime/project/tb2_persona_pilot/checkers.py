#!/usr/bin/env python3
"""Registered checkers (plan section 3/6).

Every checker: (a) is versioned and addressed by id+version+parameters,
(b) runs in a FRESH container built from an immutable snapshot of the agent's
workspace, (c) returns pass / fail / unknown -- never "the agent said so", and
(d) is cached per (checker, params, snapshot) so a probe is not re-run twice on
the same code.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dockerx
import rubric_checker
from dockerx import Mount
from rubric_checker import RubricSpec

HARNESS_FULL = "/harness_full"


@dataclass
class CheckResult:
    checker_id: str
    version: str
    parameters: Dict[str, Any]
    status: str                      # pass | fail | unknown
    reason_code: str
    evidence: Dict[str, Any]
    snapshot: str
    ran_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    evidence_ref: Optional[str] = None
    turn: Optional[int] = None

    @property
    def ref(self) -> str:
        return f"{self.checker_id}@{self.version}"

    def satisfaction(self) -> str:
        return {"pass": "met", "fail": "not_met", "unknown": "unknown"}[self.status]

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def public_summary(self) -> Dict[str, Any]:
        """What the simulator may see: verdict + measured facts, no oracle values."""
        ev = dict(self.evidence)
        if "diffs" in ev:
            ev["diffs"] = [
                {"row": d.get("row"), "period_severity": (d.get("expected") or ["?", "?"])[:2], "mismatch": True}
                for d in ev["diffs"]
            ]
        ev.pop("outputs", None)
        ev.pop("notes", None)
        if "clauses" in ev:  # rubric verdicts: keep what the user may legitimately relay
            ev["clauses"] = {
                cid: {"group": c["group"], "clause": c["clause"], "verdict": c["verdict"],
                      "agreement": c["agreement"], "reason": (c.get("reasons") or [""])[0][:200],
                      "grounded_on": [g["ref"] for g in c.get("grounding", [])][:3]}
                for cid, c in ev["clauses"].items()
            }
        return {
            "evidence_ref": self.evidence_ref,
            "checker": self.ref,
            "parameters": self.parameters,
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence": ev,
            "snapshot": self.snapshot,
        }


class CheckerRegistry:
    """id@version -> a driver mode, plus the official Terminal-Bench tests."""

    DRIVER_MODES = {
        ("entrypoint_present", "1"): "entrypoint_present",
        ("resource_budget", "1"): "resource_budget",
        ("semantic_matrix", "1"): "semantic_matrix",
        ("probe_reference_date", "1"): "probe_reference_date",
        ("clean_room_repro", "1"): "clean_room_repro",
        ("deliverable_shape", "1"): "deliverable_shape",
        ("evidence_bundle", "1"): "evidence_bundle",
        ("constraint_scan", "1"): "constraint_scan",
    }
    # checkers that are not a container driver
    EXTRA = {("official_tests", "1"), ("rubric_judge", "1")}

    def __init__(self, fixtures_dir: Path, bulk_volume: str, tests_dir: Path, run_dir: Path, judge_llm=None) -> None:
        # the model that grades LLM-judged requirements; may differ from the agent under test
        self.judge_llm = judge_llm
        self.fixtures_dir = Path(fixtures_dir)
        self.bulk_volume = bulk_volume
        self.tests_dir = Path(tests_dir)
        self.run_dir = Path(run_dir)
        self.cache: Dict[str, CheckResult] = {}
        self.results: List[CheckResult] = []

    # ---- plumbing -------------------------------------------------------
    def registered(self, checker_id: str, version: str) -> bool:
        return (checker_id, version) in self.DRIVER_MODES or (checker_id, version) in self.EXTRA

    def _mounts(self) -> List[Mount]:
        return [
            Mount(str(self.fixtures_dir), HARNESS_FULL, read_only=True),
            Mount(self.bulk_volume, "/app/logs_bulk", read_only=True),
        ]

    def _next_ref(self) -> str:
        return f"chk-{len(self.results) + 1:03d}"

    def run(
        self,
        checker_id: str,
        version: str,
        parameters: Dict[str, Any],
        snapshot_image: str,
        snapshot_digest: str,
        turn: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        if not self.registered(checker_id, version):
            raise KeyError(f"checker {checker_id}@{version} is not registered")
        evidence_key = snapshot_digest
        if (checker_id, version) == ("rubric_judge", "1"):
            # an LLM-judged requirement is graded on code AND messages, so a new
            # message invalidates the old verdict just like a new snapshot does
            evidence_key = f"{snapshot_digest}+{transcript_digest((context or {}).get('transcript', []))}"
        key = f"{checker_id}@{version}:{dockerx.params_digest(parameters)}:{evidence_key}"
        if key in self.cache:
            return self.cache[key]
        if (checker_id, version) == ("official_tests", "1"):
            result = self._official_tests(parameters, snapshot_image, snapshot_digest)
        elif (checker_id, version) == ("rubric_judge", "1"):
            result = self._rubric_judge(parameters, snapshot_image, snapshot_digest, context or {}, turn)
        else:
            result = self._driver(checker_id, version, parameters, snapshot_image, snapshot_digest)
        result.turn = turn
        result.evidence_ref = self._next_ref()
        self.cache[key] = result
        self.results.append(result)
        (self.run_dir / "checks.jsonl").open("a", encoding="utf-8").write(
            json.dumps(result.to_json(), ensure_ascii=False) + "\n"
        )
        return result

    # ---- the registered checkers ---------------------------------------
    def _driver(self, checker_id: str, version: str, parameters: Dict[str, Any], image: str, digest: str) -> CheckResult:
        mode = self.DRIVER_MODES[(checker_id, version)]
        argv = [f"python3 {HARNESS_FULL}/checker_driver.py --mode {mode}"]
        mapping = {
            "workload": "--workload",
            "reference_date": "--reference-date",
            "alt_reference_date": "--alt-reference-date",
            "cap_peak_rss_kib": "--cap-peak-rss-kib",
            "paths": "--paths",
            "workspace": "--workspace",
            "cap_wall_sec": "--cap-wall-sec",
            "deliverable": "--deliverable",
            "entrypoint_path": "--entrypoint-path",
        }
        for key, flag in mapping.items():
            if key in parameters:
                argv.append(f"{flag} {parameters[key]}")
        rc, out, err = dockerx.run_ephemeral(image, " ".join(argv), mounts=self._mounts(), network="none", timeout=2400.0)
        payload = _parse_check_line(out)
        if payload is None:
            return CheckResult(
                checker_id, version, parameters, "unknown", "driver_no_verdict",
                {"exit_code": rc, "stdout": out[-2000:], "stderr": err[-2000:]}, digest,
            )
        return CheckResult(
            checker_id, version, parameters, payload["status"], payload["reason_code"], payload.get("evidence", {}), digest
        )

    def _rubric_judge(
        self, parameters: Dict[str, Any], image: str, digest: str, context: Dict[str, Any], turn: Optional[int]
    ) -> CheckResult:
        """LLM-as-judge over the template's own rubric, grounded in the snapshot."""
        if self.judge_llm is None:
            return CheckResult("rubric_judge", "1", parameters, "unknown", "no_judge_model_configured", {}, digest)
        satisfaction = context.get("satisfaction") or {}
        statement = context.get("statement", "")
        transcript = context.get("transcript") or []
        spec = RubricSpec.from_template(
            satisfaction,
            evidence_paths=parameters.get("evidence_paths"),
            subcheck=parameters.get("subcheck"),
            samples=int(parameters.get("samples", 3)),
            message_grounded_clauses=parameters.get("message_clauses"),
        )
        bundle_res = self.run(
            "evidence_bundle", "1", {"paths": ",".join(spec.evidence_paths)} if spec.evidence_paths else {},
            image, digest, turn,
        )
        if bundle_res.status != "pass":
            return CheckResult("rubric_judge", "1", parameters, "unknown", "evidence_bundle_unavailable",
                               {"bundle": bundle_res.reason_code}, digest)
        sub_payload = None
        if spec.subcheck:
            sub = self.run(spec.subcheck["id"], spec.subcheck["version"], spec.subcheck.get("parameters", {}),
                           image, digest, turn)
            sub_payload = {"checker": sub.ref, "status": sub.status, "reason_code": sub.reason_code,
                           "evidence_ref": sub.evidence_ref, "evidence": _trim_subcheck(sub.evidence)}
        status, reason, evidence = rubric_checker.grade(
            self.judge_llm, parameters.get("requirement", "?"), statement, spec,
            bundle_res.evidence, transcript, sub_payload,
        )
        evidence["evidence_bundle_ref"] = bundle_res.evidence_ref
        evidence["judge_model"] = getattr(self.judge_llm, "model", None)
        return CheckResult("rubric_judge", "1", parameters, status, reason, evidence, digest)

    def _official_tests(self, parameters: Dict[str, Any], image: str, digest: str) -> CheckResult:
        """The original Terminal-Bench verifier, unmodified, with network on."""
        mounts = [Mount(str(self.tests_dir), "/tests", read_only=True)]
        script = "mkdir -p /logs/verifier && bash /tests/test.sh 2>&1 | tail -60; echo REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null)"
        rc, out, err = dockerx.run_ephemeral(image, script, mounts=mounts, network="bridge", timeout=2400.0)
        reward_match = re.search(r"REWARD=(\d+)", out or "")
        reward = int(reward_match.group(1)) if reward_match else None
        passed = reward == 1
        status = "pass" if passed else ("fail" if reward == 0 else "unknown")
        summary = re.findall(r"^(?:PASSED|FAILED|ERROR|=+ .*)$", out or "", re.MULTILINE)
        return CheckResult(
            "official_tests", "1", parameters, status,
            "official_tests_passed" if passed else ("official_tests_failed" if reward == 0 else "verifier_did_not_report"),
            {"reward": reward, "exit_code": rc, "tail": (out or "")[-3000:], "summary_lines": summary[-10:]},
            digest,
        )


def transcript_digest(transcript: List[Dict[str, Any]]) -> str:
    payload = [(m.get("message_id"), len(m.get("text") or "")) for m in transcript]
    return dockerx.params_digest({"t": payload})


def _trim_subcheck(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in evidence.items() if k in ("measurement", "findings", "diff_count", "same_as_delivered",
                                                       "modules_seen", "reason", "exit_code")}


def _parse_check_line(stdout: str) -> Optional[Dict[str, Any]]:
    # Split on "\n" only: splitlines() also breaks at U+0085/U+2028/U+2029, which the driver's JSON
    # (ensure_ascii=False) leaves raw when evidence quotes UTF-8-decoded binary files.
    for line in reversed((stdout or "").split("\n")):
        if line.startswith("CHECK "):
            try:
                return json.loads(line[len("CHECK ") :])
            except json.JSONDecodeError:
                return None
    return None
