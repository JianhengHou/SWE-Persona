"""Python 3.12 mini-swe-agent adapter over a harness-owned Docker container."""
import argparse
import json
import logging
import os
import shlex
import time
from pathlib import Path

import yaml
import litellm
import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from swe_persona.api_recovery import APITransportExhausted, append_audit, retry_transport


class AttachedDockerEnvironment(DockerEnvironment):
    """Reuse upstream execute; container lifetime remains owned by the harness."""
    def __init__(self, container, deadline, pid_file, timeout=600):
        self.container_id = container
        self.logger = logging.getLogger("mini.attached")
        self.config = DockerEnvironmentConfig(
            image="harness-owned", cwd="/app", timeout=timeout,
            interpreter=["bash", "-c"],
            env={"PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb"},
        )
        self.executions = []
        self.deadline = deadline
        self.pid_file = pid_file

    def execute(self, action, cwd="", *, timeout=None):
        command = action.get("command", "")
        # A question is a turn boundary, never a persistent shell or tmux channel.
        if command.strip().startswith("ask_user "):
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = []
            if len(parts) != 2:
                return {"output": "Use ask_user with exactly one quoted question.",
                        "returncode": 2, "exception_info": ""}
            raise InterruptAgentFlow({"role": "exit", "content": parts[1],
                                      "extra": {"exit_status": "AskUser", "submission": parts[1]}})
        remaining = max(1, min(self.config.timeout, int(self.deadline - time.monotonic())))
        # A dedicated process group allows host-side quiescence to stop container
        # grandchildren as well as the docker client when the episode clock expires.
        script = ("echo $$ > " + shlex.quote(self.pid_file) + "; "
                  "trap 'rm -f " + shlex.quote(self.pid_file) + "' EXIT; "
                  "timeout --kill-after=2s %ss bash -c %s > /tmp/pcb-sidecar-$$.log 2>&1 < /dev/null; "
                  "rc=$?; cat /tmp/pcb-sidecar-$$.log; rm -f /tmp/pcb-sidecar-$$.log; exit $rc"
                  % (remaining, shlex.quote(command)))
        wrapped = dict(action, command="setsid bash -c " + shlex.quote(script))
        result = super().execute(wrapped, cwd, timeout=remaining + 5)
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in result["output"]:
            result["output"] += (
                "\nSubmission was NOT accepted: issue echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT "
                "as a separate action with no preceding output. Do not use a no-op or poll.\n"
            )
        self.executions.append({"command": command, **result})
        return result

    def cleanup(self):
        pass


def turn_reply(end, interactive):
    extra = end.get("extra", {})
    reply = extra.get("submission") or end.get("content") or ""
    if interactive and extra.get("exit_status") != "AskUser" and not reply.strip():
        raise ValueError("A review submission must include a user-facing handoff.")
    return reply or "Patch submitted for review."


def save_trajectory(agent, trajectory):
    temporary = trajectory.with_suffix(".tmp")
    agent.save(temporary)
    os.replace(temporary, trajectory)


class RecipientTextModel(LitellmTextbasedModel):
    # Retrying a rejected request with identical input cannot repair it.
    abort_exceptions = LitellmTextbasedModel.abort_exceptions + [litellm.BadRequestError, APITransportExhausted]

    def _query(self, messages, **kwargs):
        started = time.monotonic()
        def request(seconds):
            return super(RecipientTextModel, self)._query(messages, **dict(kwargs,timeout=seconds,num_retries=0))
        response = retry_transport(request,request_timeout=float(os.environ.get('SWE_PERSONA_AGENT_HTTP_TIMEOUT',300)),
            deadline=self.recovery_deadline,
            audit=lambda record:append_audit(self.recovery_audit,dict(record,role='agent')))
        def field(value, name, default=None):
            return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)
        choices = field(response, 'choices', []) or []
        choice = choices[0] if choices else {}
        message = field(choice, 'message', {}) or {}
        usage = field(response, 'usage', {}) or {}
        details = field(usage, 'completion_tokens_details', {}) or {}
        text = field(message, 'content', '') or ''
        reasoning = field(message, 'reasoning_content', '') or ''
        append_audit(self.output_audit, {
            'role':'agent','model':field(response,'model',self.config.model_name),
            'elapsed_sec':time.monotonic()-started,
            'input_tokens':field(usage,'prompt_tokens'),
            'output_tokens':field(usage,'completion_tokens'),
            'requested_max_tokens':self.output_budget,
            'effort':self.output_effort or None,
            'finish_reason':field(choice,'finish_reason'),
            'reasoning_tokens':field(details,'reasoning_tokens'),
            'reasoning_content_chars':len(reasoning) if isinstance(reasoning,str) else None,
            'visible_output_chars':len(text) if isinstance(text,str) else None,
        })
        return response


def make_model(request, config):
    return RecipientTextModel(
        model_name="openai/" + request["model"].removeprefix("openai/"),
        model_kwargs={"api_base": request["api_base"], "timeout": min(600, request["timeout"]), "num_retries": 0, **json.loads(os.environ.get("SWE_PERSONA_AGENT_REQUEST_OPTIONS", "{}"))},
        observation_template=config["model"]["observation_template"],
        format_error_template=config["model"]["format_error_template"],
        cost_tracking="ignore_errors",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    directory = Path(request["directory"])
    trajectory = directory / "trajectory.json"
    config = yaml.safe_load((Path(minisweagent.__file__).parent / "config/default.yaml").read_text())
    system = config["agent"]["system_template"] + "\n" + request["task_brief"]
    instance = config["agent"]["instance_template"]
    if request["interactive"]:
        system += request.get("interactive_note", "") + (
            "\nThe requester is available. To ask a question, your sole bash action must be "
            "ask_user followed by one shell-quoted question. It yields to the user simulator. "
            "Do not poll for a reply. A submission yields your current patch for review; "
            "subsequent user feedback may require further work in the same workspace.\n"
            "Only the text in ask_user's quoted argument or AFTER the submission marker "
            "is delivered to the requester. Your THOUGHT and shell output are NOT delivered. "
            "Include your full user-facing response, evidence summary and any requested "
            "handoff sections in that argument or submission text. To submit, use "
            "printf '%s\\n' COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT 'your complete handoff'. "
            "A bare submission marker is insufficient. Do not expose internal reasoning. "
            "Preparing the patch, tests, and isolated preview/undo checks is authorized; "
            "request confirmation only for an action the requester actually restricted.\n"
        )
        instance = instance.replace(
            "After this command, you cannot continue working on this task.",
            "This submits the current patch for review; later user feedback may require continuation.",
        )
    model = make_model(request, config)
    env = AttachedDockerEnvironment(request["container"],
                                    time.monotonic() + request["timeout"], request["pid_file"])
    model.recovery_deadline = env.deadline
    model.recovery_audit = directory / 'transport-recovery.jsonl'
    model.output_audit = directory / 'api-calls.jsonl'
    model.output_budget = int(os.environ['SWE_PERSONA_AGENT_OUTPUT_BUDGET'])
    model.output_effort = os.environ.get('SWE_PERSONA_AGENT_REASONING_EFFORT','')
    agent = DefaultAgent(model, env, system_template=system,
                         instance_template=instance, cost_limit=0,
                         step_limit=0, output_path=None)
    if trajectory.exists():
        saved = json.loads(trajectory.read_text())
        agent.messages = [m for m in saved["messages"] if m.get("role") != "exit"]
        agent.n_calls = saved["info"]["model_stats"]["api_calls"]
        agent.cost = saved["info"]["model_stats"]["instance_cost"]
        agent.add_messages(model.format_message(role="user", content=request["user_message"]))
    else:
        agent.extra_template_vars = {"task": request["user_message"]}
        agent.add_messages(
            model.format_message(role="system", content=agent._render_template(system)),
            model.format_message(role="user", content=agent._render_template(instance)),
        )
    while True:
        try:
            agent.step()
            agent.n_consecutive_format_errors = 0
        except FormatError as exc:
            agent.add_messages(*exc.messages)
            agent.n_consecutive_format_errors += 1
            if agent.n_consecutive_format_errors >= 3:
                raise RuntimeError("mini-swe-agent repeated format errors")
        except InterruptAgentFlow as exc:
            agent.add_messages(*exc.messages)
        except APITransportExhausted as exc:
            # Preserve the workspace/trajectory; the episode will freeze and grade it.
            (directory / ("turn-%s.json" % request["turn"])).write_text(json.dumps({
                'reply':'', 'status':'api_interrupted', 'executions':env.executions,
                'api_error':str(exc), 'mini_version':minisweagent.__version__}))
            return
        finally:
            save_trajectory(agent, trajectory)
        if agent.messages[-1].get("role") == "exit":
            end = agent.messages[-1]
            extra = end.get("extra", {})
            try:
                reply = turn_reply(end, request["interactive"])
            except ValueError:
                agent.messages.pop()
                agent.add_messages(model.format_message(
                    role="user", content="Submission not delivered: include your complete user-facing "
                    "response after the submission marker, including requested handoff sections. "
                    "Text in THOUGHT or shell output is not visible to the requester."))
                save_trajectory(agent, trajectory)
                continue
            payload = {"reply": reply,
                       "status": "working" if extra.get("exit_status") == "AskUser" else "done",
                       "executions": env.executions,
                       "mini_version": minisweagent.__version__}
            (directory / ("turn-%s.json" % request["turn"])).write_text(json.dumps(payload))
            return


if __name__ == "__main__":
    main()
