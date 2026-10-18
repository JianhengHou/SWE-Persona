#!/usr/bin/env python3
"""Versioned, snapshot-isolated checkers for the SWE-bench Pro pilot."""

from __future__ import annotations

import json
import hashlib
import shlex
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dockerx
import rubric_checker
from dockerx import Mount
from llm import SchemaRepairFailed
from rubric_checker import RubricSpec
from evidence_policy import EVIDENCE_POLICY

HARNESS_FULL = "/harness_full"


@dataclass
class CheckResult:
    checker_id: str
    version: str
    parameters: Dict[str, Any]
    status: str
    reason_code: str
    evidence: Dict[str, Any]
    snapshot: str
    ran_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    evidence_ref: Optional[str] = None
    turn: Optional[int] = None
    duration_sec: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.checker_id}@{self.version}"

    def satisfaction(self) -> str:
        return {"pass": "met", "fail": "not_met", "unknown": "unknown"}[self.status]

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def public_summary(self) -> Dict[str, Any]:
        evidence = _public_evidence(self.evidence)
        if self.evidence.get("error_category"):
            evidence = {"error_category": self.evidence["error_category"],
                        "agent_revision_allowed": False,
                        "note": "Evaluator malfunction; not an agent violation or request to repair code."}
        if "clauses" in evidence:
            evidence["clauses"] = {
                clause_id: {
                    "group": clause["group"],
                    "clause": clause["clause"],
                    "verdict": clause["verdict"],
                    "agreement": clause["agreement"],
                    "reason": (clause.get("reasons") or [""])[0][:200],
                    "grounded_on": [item["ref"] for item in clause.get("grounding", [])][:3],
                }
                for clause_id, clause in evidence["clauses"].items()
            }
        return {
            "evidence_ref": self.evidence_ref,
            "checker": self.ref,
            "parameters": self.parameters,
            "status": self.status,
            "reason_code": self.reason_code,
            "evidence": evidence,
            "snapshot": self.snapshot,
        }


def _public_evidence(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Remove checker-only material, including nested rubric subchecks."""
    evidence = dict(raw)
    for key in (
        "files", "listing", "failed_tests", "runner_tail", "apply_error",
        "status_lines", "test_overlay", "judge_audit", "stdout", "stderr", "reference_control",
    ):
        evidence.pop(key, None)
    subcheck = evidence.get("subcheck")
    if isinstance(subcheck, dict):
        public_subcheck = dict(subcheck)
        nested = public_subcheck.get("evidence")
        if isinstance(nested, dict):
            public_subcheck["evidence"] = _public_evidence(nested)
        evidence["subcheck"] = public_subcheck
    return evidence


class CheckerRegistry:
    DRIVER_MODES = {
        ("task_patch_manifest", "1"): "task_patch_manifest",
        ("isolated_patch", "1"): "isolated_patch",
        ("evidence_bundle", "1"): "evidence_bundle",
    }
    EXTRA = {("official_tests", "1"), ("rubric_judge", "1")}

    def __init__(
        self,
        assets_dir: Path,
        run_dir: Path,
        base_commit: str,
        judge_llm=None,
        rubric_checker_enabled: bool = True,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.run_dir = Path(run_dir)
        self.base_commit = base_commit
        self.judge_llm = judge_llm
        self.rubric_checker_enabled = rubric_checker_enabled
        self.cache: Dict[str, CheckResult] = {}
        self.results: List[CheckResult] = []
        self.command_records: List[Dict[str, Any]] = []

    def registered(self, checker_id: str, version: str) -> bool:
        return (checker_id, version) in self.DRIVER_MODES or (checker_id, version) in self.EXTRA

    def _mounts(self, official: bool = False) -> List[Mount]:
        """File allowlist: public checks never mount the private assets directory."""
        names = ["checker_driver.py"]
        mounts = []
        if official:
            names += ["task_spec.json", "test.patch", "gold.patch", "run_script.sh", "parser.py"]
        else:
            raw = json.loads((self.assets_dir / "task_spec.json").read_text())
            public = {k: raw.get(k) for k in ("problem_statement", "requirements", "interface")}
            path = self.run_dir / ".public-checker-contract.json"
            path.write_text(json.dumps(public, ensure_ascii=False))
            mounts.append(Mount(str(path), f"{HARNESS_FULL}/task_spec.json", read_only=True))
        mounts.extend(Mount(str(self.assets_dir / name), f"{HARNESS_FULL}/{name}", read_only=True)
                      for name in names)
        return mounts

    def record_agent_turn(self, turn, steps, snapshot):
        """Host receipts; snapshot_after_turn is not a per-command code identity."""
        for step in steps:
            record = {"turn": turn, "index": step.index, "command": step.command,
                      "exit_code": step.exit_code, "stdout": step.stdout,
                      "output_capture": "full-host-output-v1",
                      "snapshot_after_turn": snapshot}
            self.command_records.append(record)
            with (self.run_dir / "command_receipts.jsonl").open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _command_evidence(self, transcript=()):
        index, recent = [], []
        for number, record in enumerate(self.command_records, 1):
            digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            item = {"step": number, "turn": record["turn"], "exit_code": record["exit_code"],
                    "command": record["command"], "command_truncated": False,
                    "full_record_sha256": digest, "snapshot_after_turn": record["snapshot_after_turn"]}
            index.append(item)
            if number > len(self.command_records) - 8:
                recent.append(dict(item, stdout_tail=record["stdout"][-2000:],
                                   output_truncated=len(record["stdout"]) > 2000))
        files = []
        for name, data in (("commands.index", index), ("commands.recent", recent)):
            text = json.dumps({"scope": "Host-observed commands, not necessarily tests. Turn-end snapshot "
                              "does not prove a command checked the final code. Omitted output is not absence.",
                              "records": data}, ensure_ascii=False)
            files.append({"path": "host:" + name, "text": text, "truncated": False,
                          "sha256": hashlib.sha256(text.encode()).hexdigest()})
        # Retrieve from host-retained records only: never open an agent-supplied
        # path or treat a claim about a historical implementation as evidence.
        replies = [m.get("text", "") for m in transcript if m.get("role") == "agent"]
        latest = replies[-1] if replies else ""
        symbols = set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", latest))
        symbols.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", latest))
        terms = sorted(symbols)[:32]
        matches, seen, remaining = [], set(), 16000
        eligible = []
        frequencies = {term: 0 for term in terms}
        for number, record in enumerate(self.command_records, 1):
            output = record.get("stdout", "")
            hits = [term for term in terms if re.search(r"\b" + re.escape(term) + r"\b", output)]
            if hits:
                for term in hits:
                    frequencies[term] += 1
                eligible.append((number, record, hits))
        # Prefer complete earlier outputs over snippets of commands writing code.
        # Rare referenced symbols outrank generic tokens in recent outputs.
        ranked = sorted(eligible, key=lambda item: (
            max(1 / frequencies[term] for term in item[2]),
            sum(1 / frequencies[term] for term in item[2]), item[0]), reverse=True)
        for number, record, hits in ranked:
            output = record.get("stdout", "")
            digest = hashlib.sha256(output.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            limit = min(8000, remaining)
            if not limit or len(matches) >= 4:
                break
            excerpt = output[:limit]
            matches.append({"step": number, "turn": record["turn"],
                            "matched_symbols": hits,
                            "exit_code": record["exit_code"], "stdout": excerpt,
                            "stdout_sha256": digest, "output_truncated": len(output) > limit,
                            "command": record["command"],
                            "command_truncated": False,
                            "snapshot_after_turn": record["snapshot_after_turn"]})
            remaining -= len(excerpt)
        if terms:
            text = json.dumps({"scope": "Retrieved host observations of earlier commands, not necessarily "
                               "baseline code or final-code verification. Output may itself contain agent-authored "
                               "content. Turn-end snapshots are not per-command identities. Missing or truncated "
                               "retrieval is insufficient evidence, not proof of fabrication.",
                               "query_symbols": terms, "matching_records": len(eligible),
                               "selected_records": len(matches), "records": matches}, ensure_ascii=False, indent=2)
            files.append({"path": "host:commands.history", "text": text, "truncated": False,
                          "sha256": hashlib.sha256(text.encode()).hexdigest()})
        return files

    def _complete_command_evidence(self):
        """Preserve full command evidence while bounding oversized stdout views.

        Legacy receipts can already contain a trim marker. Preserve it and never
        claim the missing bytes or per-command snapshot identity were recovered.
        """
        files = []
        total_output = sum(len(record.get("stdout", "")) for record in self.command_records)
        output_limit = (max(1, min(65536, 120000 // max(1, len(self.command_records))))
                        if total_output > 120000 else 65536)
        for number, record in enumerate(self.command_records, 1):
            output = record.get("stdout", "")
            output_omitted = len(output) > output_limit
            if output_omitted:
                head = output_limit // 2
                tail = output_limit - head
                output_view = (output[:head] +
                    f"\n[... {len(output) - output_limit} host stdout characters omitted from judge view; "
                    "complete raw receipt remains retained. Omission is not evidence of absence. ...]\n" +
                    output[-tail:])
            else:
                output_view = output
            payload = {
                "evidence_policy": EVIDENCE_POLICY,
                "scope": "Host-observed command and retained output. Agent-authored output is not "
                         "independent proof. snapshot_after_turn is a turn-end identity, not a "
                         "per-command identity; it cannot alone establish final-code validation. "
                         "Every command string is complete. Oversized stdout views explicitly mark "
                         "omissions; unseen output cannot support absence or success claims.",
                "step": number,
                "full_record_sha256": hashlib.sha256(
                    json.dumps(record, sort_keys=True).encode()).hexdigest(),
                "command_truncated": False,
                "retained_output_truncated": output_omitted,
                "original_capture_may_be_incomplete": record.get("output_capture") != "full-host-output-v1",
                "original_trim_marker": bool(re.search(r"\.\.\.\[\d+ chars trimmed\]\.\.\.", record.get("stdout", ""))),
                **record,
                "stdout": output_view,
                "stdout_view_policy": "host-stdout-head-tail-budget-v1",
                "stdout_view_truncated": output_omitted,
                "stdout_full_characters": len(output),
                "stdout_full_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "complete_raw_receipt_retained": True,
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            files.append({"path": f"host:command-{number:03d}", "text": text,
                          "truncated": output_omitted, "sha256": hashlib.sha256(text.encode()).hexdigest()})
        return files

    def _public_submission_protocol(self):
        """Only the public submission block actually delivered in the host trace.

        Never read agent-supplied paths or send trajectory configuration/secrets.
        This supplies evidence of task necessity, not a scope-rule exemption.
        """
        path = self.run_dir / "minisweagent/trajectory.json"
        if not path.is_file():
            return []
        trajectory = json.loads(path.read_text())
        opening = next((m.get("content", "") for m in trajectory.get("messages", [])
                        if m.get("role") == "user"), "")
        prefix = "When you've completed your work, you MUST submit your changes as a git patch."
        start = opening.rfind(prefix)
        end = opening.find("</instructions>", start)
        if start < 0 or end < start:
            return []
        text = json.dumps({
            "scope": "Public submission instructions actually supplied in the initial agent input. "
                     "They establish the required delivery workflow, not code correctness. "
                     "Evaluate necessary artifacts and the actual delivered patch against the rubric; "
                     "workspace presence alone does not establish inclusion in the delivered patch.",
            "source": "host-retained-mini-trajectory.initial-user-message",
            "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "opening_sha256": hashlib.sha256(opening.encode()).hexdigest(),
            "submission_instructions": opening[start:end].strip(),
            "initial_system_instructions": [m.get("content", "")
                for m in trajectory.get("messages", []) if m.get("role") == "system"],
        }, ensure_ascii=False, indent=2)
        return [{"path": "host:public-submission-protocol", "text": text, "truncated": False,
                 "sha256": hashlib.sha256(text.encode()).hexdigest()}]

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
        if checker_id == "rubric_judge":
            evidence_key += "+" + transcript_digest((context or {}).get("transcript", []))
            evidence_key += "+" + dockerx.params_digest({"protocol": rubric_checker.GROUNDING_PROTOCOL,
                "clause_protocol": rubric_checker.CLAUSE_PROTOCOL,
                "context": context or {}, "commands": self.command_records,
                "model": getattr(self.judge_llm, "model", None)})
        key = f"{checker_id}@{version}:{dockerx.params_digest(parameters)}:{evidence_key}"
        if key in self.cache:
            return self.cache[key]
        started = time.time()
        if checker_id == "rubric_judge" and not self.rubric_checker_enabled:
            # This is an explicitly non-scoring timing trial.  Do not collect the
            # rubric evidence bundle or run its deterministic subcheck: both exist
            # solely to support the LLM rubric verdict and would skew the timing.
            result = CheckResult(
                "rubric_judge",
                "1",
                parameters,
                "unknown",
                "disabled_for_timing_trial",
                {"rubric_checker_enabled": False},
                snapshot_digest,
            )
        elif checker_id == "official_tests":
            result = self._official_tests(parameters, snapshot_image, snapshot_digest)
        elif checker_id == "rubric_judge":
            result = self._rubric_judge(parameters, snapshot_image, snapshot_digest, context or {}, turn)
        else:
            result = self._driver(checker_id, version, parameters, snapshot_image, snapshot_digest)
        result.duration_sec = round(time.time() - started, 3)
        result.turn = turn
        result.evidence_ref = self._next_ref()
        if result.status == "unknown" and result.reason_code in (
            "driver_no_verdict", "checker_error", "evidence_bundle_unavailable", "unsafe_evidence_path"
        ):
            result.evidence["error_category"] = "evidence_collection_error"
        if result.status == "unknown" and result.reason_code in (
            "judge_output_invalid", "judge_response_unavailable", "no_judge_model_configured"
        ):
            result.evidence["error_category"] = "judge_error"
        self.cache[key] = result
        self.results.append(result)
        with (self.run_dir / "checks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")
        return result

    def _driver(self, checker_id: str, version: str, parameters: Dict[str, Any], image: str, digest: str) -> CheckResult:
        mode = self.DRIVER_MODES[(checker_id, version)]
        argv = ["python3", f"{HARNESS_FULL}/checker_driver.py", "--mode", mode,
                "--base-commit", str(parameters.get("base_commit", self.base_commit))]
        if "allowed_paths" in parameters:
            argv += ["--allowed-paths", json.dumps(parameters["allowed_paths"])]
        if "paths" in parameters:
            argv += ["--paths", str(parameters["paths"])]
        script = " ".join(shlex.quote(item) for item in argv)
        rc, out, err = dockerx.run_ephemeral(image, script, mounts=self._mounts(), network="none", timeout=3900.0)
        payload = _parse_check_line(out)
        if payload is None:
            return CheckResult(checker_id, version, parameters, "unknown", "driver_no_verdict",
                               {"exit_code": rc, "stdout": out[-2000:], "stderr": err[-2000:]}, digest)
        return CheckResult(checker_id, version, parameters, payload["status"], payload["reason_code"],
                           payload.get("evidence", {}), digest)

    def _official_tests(self, parameters: Dict[str, Any], image: str, digest: str) -> CheckResult:
        script = (
            f"python3 {HARNESS_FULL}/checker_driver.py --mode official_tests "
            f"--base-commit {shlex.quote(self.base_commit)}"
        )
        rc, out, err = dockerx.run_ephemeral(image, script, mounts=self._mounts(official=True), network="none", timeout=3900.0)
        payload = _parse_check_line(out)
        if payload is None:
            return CheckResult("official_tests", "1", parameters, "unknown", "driver_no_verdict",
                               {"exit_code": rc, "stdout": out[-2000:], "stderr": err[-2000:]}, digest)
        return CheckResult("official_tests", "1", parameters, payload["status"], payload["reason_code"],
                           payload.get("evidence", {}), digest)

    def _rubric_judge(
        self, parameters: Dict[str, Any], image: str, digest: str,
        context: Dict[str, Any], turn: Optional[int],
    ) -> CheckResult:
        if self.judge_llm is None:
            return CheckResult("rubric_judge", "1", parameters, "unknown", "no_judge_model_configured", {}, digest)
        spec = RubricSpec.from_template(
            context.get("satisfaction") or {},
            evidence_paths=parameters.get("evidence_paths"),
            subcheck=parameters.get("subcheck"),
            samples=int(parameters.get("samples", 3)),
            message_grounded_clauses=parameters.get("message_clauses"),
        )
        spec.message_clause_requires_artifact = parameters.get("message_clause_requires_artifact", [])
        from evidence_policy import activation_context
        spec.activation_context = activation_context(context.get("introduced_turn"),
                                                     context.get("transcript") or [])
        bundle_params = {
            "base_commit": self.base_commit,
            "paths": ",".join(spec.evidence_paths),
        }
        bundle = self.run("evidence_bundle", "1", bundle_params, image, digest, turn)
        if bundle.status != "pass":
            return CheckResult("rubric_judge", "1", parameters, "unknown", "evidence_bundle_unavailable",
                               {"bundle": bundle.reason_code}, digest)
        sub_payload = None
        if spec.subcheck:
            if (spec.subcheck["id"], spec.subcheck["version"]) not in self.DRIVER_MODES:
                return CheckResult("rubric_judge", "1", parameters, "unknown", "evidence_bundle_unavailable",
                                   {"bundle": "non_public_subcheck_forbidden"}, digest)
            sub = self.run(spec.subcheck["id"], spec.subcheck["version"], spec.subcheck.get("parameters", {}),
                           image, digest, turn)
            if sub.evidence.get("error_category"):
                return CheckResult("rubric_judge", "1", parameters, "unknown", "evidence_bundle_unavailable",
                                   {"bundle": "subcheck_collection_error", "subcheck_ref": sub.evidence_ref}, digest)
            sub_payload = {
                "checker": sub.ref,
                "status": sub.status,
                "reason_code": sub.reason_code,
                "evidence_ref": sub.evidence_ref,
                "evidence": {key: value for key, value in sub.evidence.items()
                             if key in ("changed_paths", "out_of_scope", "required_count", "passed_required_count",
                                        "failed_required_count", "head", "base_commit", "initial_snapshot",
                                        "delivery_manifest", "failure_case_records")},
            }
        evidence_bundle = dict(bundle.evidence)
        host_files = self._complete_command_evidence() if self.command_records else []
        host_files += self._public_submission_protocol()
        evidence_bundle["files"] = list(bundle.evidence.get("files", [])) + host_files
        try:
            status, reason, evidence = rubric_checker.grade(
                self.judge_llm,
                parameters.get("requirement", "?"),
                context.get("statement", ""),
                spec,
                evidence_bundle,
                context.get("transcript") or [],
                sub_payload,
            )
        except rubric_checker.JudgeOutputError as exc:
            return CheckResult("rubric_judge", "1", parameters, "unknown", "judge_output_invalid",
                               {"grounding_protocol": rubric_checker.GROUNDING_PROTOCOL,
                                "clause_protocol": rubric_checker.CLAUSE_PROTOCOL,
                                "judge_audit": exc.audit, "invalid_grounding_refs": exc.invalid_refs}, digest)
        except (SchemaRepairFailed, RuntimeError) as exc:
            return CheckResult(
                "rubric_judge", "1", parameters, "unknown", "judge_response_unavailable",
                {"error": f"{type(exc).__name__}: {exc}"[:500],
                 "judge_model": getattr(self.judge_llm, "model", None)},
                digest,
            )
        evidence["evidence_bundle_ref"] = bundle.evidence_ref
        evidence["judge_model"] = getattr(self.judge_llm, "model", None)
        return CheckResult("rubric_judge", "1", parameters, status, reason, evidence, digest)


def transcript_digest(transcript: List[Dict[str, Any]]) -> str:
    payload = [(item.get("message_id"), item.get("status"), item.get("text")) for item in transcript]
    return dockerx.params_digest({"transcript": payload})


def _parse_check_line(stdout: str) -> Optional[Dict[str, Any]]:
    # Split on "\n" only: splitlines() also breaks at U+0085/U+2028/U+2029, which the driver's JSON
    # (ensure_ascii=False) leaves raw when evidence quotes UTF-8-decoded binary files.
    for line in reversed((stdout or "").split("\n")):
        if line.startswith("CHECK "):
            try:
                payload = json.loads(line[6:])
                if (not isinstance(payload, dict) or payload.get("status") not in ("pass", "fail", "unknown")
                        or not isinstance(payload.get("reason_code"), str)
                        or not isinstance(payload.get("evidence", {}), dict)):
                    return None
                return payload
            except json.JSONDecodeError:
                return None
    return None
