# SWE-Persona Lite

SWE-Persona evaluates coding models on software tasks with a simulated user whose requirements and interaction behavior depend on a frozen persona. This distribution contains **100 task/persona pairs**, one Parquet file, and a runnable API evaluation harness built on Harbor and Docker.

| Source benchmark | Tasks |
| --- | ---: |
| Terminal-Bench 3.0 | 23 |
| SWE-bench Pro | 26 |
| SWE-bench-Live MultiLang | 19 |
| SWE-Together | 4 |
| DeepSWE | 28 |
| **Total** | **100** |

The Lite selection contains the same 100 task/persona pairs; public profile names and related metadata have been standardized. This package contains no Full dataset, historical evaluation runs, internal test reports, or task-generation scripts. The required profile runtimes, prompts, calibration fixtures and license notices remain under `src/swe_persona/vendor/`; they are executable dependencies. Docker image layers are downloaded or built on the evaluator's machine.

## 1. Install

Use Linux, Python 3.12, Docker Engine with the Compose plugin, and permission to run Docker. The harness pins Harbor 0.23.0 and mini-swe-agent 2.4.6. Image downloads and builds require network access and sufficient disk space. Task CPU, memory, storage and GPU requirements are recorded in Parquet.

The SWE-Pro mini network-probe cache reads the host's Linux `boot_id` when available. On macOS, it uses `sysctl -n kern.boottime` with a five-second timeout instead. If neither provides a reboot token, caching is disabled and the Docker kernel probe still runs; missing host metadata alone does not block startup. A failed kernel probe still blocks execution under the required network policy. This cache compatibility does not certify all profiles or tasks on macOS.

Install [Git LFS](https://git-lfs.com/) before cloning. The Lite Parquet is stored in Git LFS; use the commands below to download the actual dataset and verify the release files.

```bash
git lfs install
git clone https://github.com/WillDreamer/SWE-Persona.git
cd SWE-Persona
git lfs pull
sha256sum -c SHA256SUMS
python3.12 -m venv .venv
source .venv/bin/activate
pip install -c requirements.lock.txt .
docker version
docker compose version
swe-persona list
```

Run these commands from this directory. The CLI defaults to `swe-persona-lite.parquet`; elsewhere, pass an absolute dataset path before the subcommand:

```bash
swe-persona --dataset /path/to/SWE-Persona/swe-persona-lite.parquet list
```

Installation does not require the author's original repositories or integration directory. `requirements.lock.txt` is the tested dependency set used as pip constraints. `SHA256SUMS` and `release.json` identify the distributed files and runtime.

## 2. Configure the agent and user-simulator APIs

Configure only two API connections: `agent` and `user_simulator`. Persona interaction and grading both use the `user_simulator` model, endpoint, API key and request options:

The default user simulator is **`gpt-6-luna` with reasoning effort `medium`**, using the Azure Responses endpoint `https://matraix.services.ai.azure.com/openai/v1/responses`. This default is independent of the coding agent model. Keep it fixed when evaluating different coding agents. Persona judges share this same simulator configuration. If `user_simulator` is omitted from the input JSON, the loader supplies this default; an explicit section overrides it.

| Configuration section | Responsibility |
| --- | --- |
| `agent` | The coding model being evaluated; produces actions and replies to the user. |
| `user_simulator` | Provides persona interaction and evidence-based persona grading through the same API. Native tests run separately in Docker. |

```bash
cp configs/api.example.json configs/api.local.json
export SWE_PERSONA_AGENT_API_KEY='YOUR_AGENT_API_KEY'
export SWE_PERSONA_SIMULATOR_API_KEY='YOUR_SIMULATOR_API_KEY'
```

Edit the `agent` section of `configs/api.local.json` to select the coding model and its provider. The example already configures the default user simulator. `base_url` is the API root, such as `https://provider.example/v1` or `https://matraix.services.ai.azure.com/openai/v1`, without `/chat/completions` or `/responses`. `api_key_env` names an environment variable; it does not contain the key. Do not put credentials in JSON or Parquet. The two variables may hold the same key when one provider serves both connections. No separate judge key or endpoint is required.

An example role configuration is:

```json
{
  "api": "chat_completions",
  "base_url": "https://provider.example/v1",
  "model": "YOUR_MODEL",
  "api_key_env": "SWE_PERSONA_AGENT_API_KEY",
  "strict_json_schema": true,
  "structured_output_mode": "json_schema",
  "max_tokens_field": "max_completion_tokens",
  "timeout_sec": 120
}
```

The input JSON accepts only `agent` and `user_simulator`; `agent` is required and `user_simulator` can be omitted to use the default above. The generic example uses Chat Completions for the coding agent and Responses for the simulator. The `swe-bench-pro-mini` coding agent specifically requires Chat Completions. Keep `user_simulator` fixed when comparing coding models; change only `agent`.

Interaction and grading remain separate model calls with separate prompts, allowed evidence and audit records. The simulator's approval is not itself a passing score. The loader creates an internal `judge` alias from `user_simulator` for the profile adapters; this alias may appear in `run-config.json` and logs, but is not a third API configuration. Adding a separate `judge` section to the input configuration is rejected.

Provider options:

- `structured_output_mode`: `json_schema` requests schema-constrained output; `json_object` requests JSON mode; `prompt` supplies the schema in instructions without sending a provider `response_format`. Local validation still applies. Use `prompt` for providers that cannot enforce the schema. Setting only `strict_json_schema: false` selects `json_object`, not `prompt`.
- `structured_output_fallback`: defaults to `true`. If the provider explicitly rejects the selected output mode as unsupported, the request automatically moves through `json_schema` → `json_object` → `prompt`, starting at the configured mode. There are at most two downgrades. Every attempt shares the original deadline, token budget and any profile call limit; it never repeats a tool action. Rejected modes are remembered in the worker process for the same endpoint, model, API transport and initial mode. Authentication errors, rate limits, timeouts, invalid schema definitions and malformed model output do not trigger this capability fallback. Local schema validation and bounded format repair remain active at every level. Set this option to `false` to require the configured provider format and surface its rejection immediately. API audits record both configured and effective modes, and each downgrade. DeepSWE uses the same policy; its corresponding host variable is `<ROLE>_STRUCTURED_OUTPUT_FALLBACK=true|false`.
- `max_tokens_field`: for Chat Completions, select `max_completion_tokens` or `max_tokens` according to the provider. A positive integer limit in `request_options` overrides the profile's per-call output limit. Responses uses `max_output_tokens` instead. Reasoning models may consume this output budget before producing a visible answer.
- `request_options`: provider-supported settings such as `temperature`, `reasoning_effort` and `extra_body`. The shared `reasoning_effort` option is mapped to `reasoning.effort` for Responses. Omit unsupported options. DeepSWE supports `reasoning_effort`, `temperature`, `top_p`, token limits, `seed`, `stop` and `extra_body`.
- `timeout_sec`: the API client's request timeout; some profiles supply their own per-call timeout. It is separate from the task's total time budget.

Completion recovery uses `bounded-output-recovery-v1` in the public role API (also used by SWE-Together) and DeepSWE's host API. Every coding-agent call with no explicit output limit starts at 8,192 tokens for `none`/`minimal`, 16,384 for `low`, 65,536 for `medium`/`high`, and 131,072 for `xhigh`/`max`, whether its response is schema-constrained JSON or adapter-parsed text. These are harness defaults, not provider capacity guarantees. Pro shell, Pro mini, SWE-Live, SWE-Together, Terminal-Bench 3 and DeepSWE all use this common resolver; Pro mini receives the resolved numeric limit from the host instead of using a provider default.

`length`, `max_tokens`, `max_completion_tokens`, and `max_output_tokens` all identify truncation. An automatically sized call can move to the next budget step once, up to 131,072 tokens. This retries the same request before any action executes and does not consume a format-repair attempt. The increase allowance and absolute deadline are retained through subsequent format repairs. Every generated attempt and known token usage is recorded. Truncated structured responses are rejected even when their visible text parses as JSON.

When coding-agent effort is omitted, the harness defaults its output budget to 65,536 tokens. This budget fallback does not set the provider’s reasoning effort. Simulator and judge calls do not use coding-agent effort sizing: they retain their profile per-call fallbacks unless overridden by configuration (the default user-simulator configuration sets 32,768). This adapter change requires fresh calibration and new attempts.

An explicit token limit in `request_options` remains a hard ceiling and overrides the adapter's per-call limit. Without a configured override, an explicit per-call limit is also a hard ceiling. For example, `agent.request_options.max_completion_tokens=65536` prevents growth above 65,536; omit that setting to enable automatic growth in a caller that supplies no explicit limit. A provider HTTP 400 identifying an excessive token limit permits at most four decreases; a reported smaller provider ceiling is remembered by the client. Credentials, unsupported schemas and unrelated parameter errors are not retried by this policy. Configurations and schemas are never modified or silently downgraded.

Request timeouts stay at their configured values. Transport retries, token-budget adjustments and format repairs share an absolute deadline, further bounded by the caller and episode limits where supplied. The default recovery window is three configured request timeouts plus three seconds of transport backoff; a retry never restarts that window. Simulator and judge calls retain their configured budgets and do not receive effort-based automatic increases.

Successful coding-agent API audit records include the requested output limit, effective effort, finish reason, reasoning-token count when the provider reports it, and visible-output character count. Missing provider usage details remain `null`; they are never inferred from hidden reasoning text.

All six runtime profiles use `simulator-json-envelope-v1`: simulator system prompts start with an explicit JSON output protocol, followed by the original persona and task instructions. Every required field must be present, including empty strings, empty arrays and `false` when applicable. Key order is unrestricted. User messages and private audit rationale belong in their designated fields; no prose or Markdown fences may appear outside the JSON object.

Examples are validated against the current response schema and use current allowed actions, document IDs and observation values. They illustrate structure and cannot grant permissions, disclose future requirements or approve task completion. DeepSWE's document-selection recovery shows empty `message` and `closing`; SWE-Together's acceptance fields appear only in its acceptance example when the current completion gates permit it. Existing schema validation, controller checks, format repair and recoverable judge handling remain active. The protocol applies to prompts loaded from existing Parquet rows as well as newly generated prompts. Changed prompts require a fresh runtime fingerprint and calibration for new scored attempts.

The JSON action agents for DeepSWE, SWE-Together and Terminal-Bench 3 use `coding-agent-json-envelope-v1`. Their system prompts start with an explicit response protocol and a section for each action field. All required fields must be present; key order is unrestricted. User-facing plans, questions and handoffs belong inside the designated JSON string. Examples are checked against each runtime's existing action schema, contain no task-specific file names or feature requests, and cannot introduce permissions or an additional approval step. Completion actions remain claims for user/controller review, not declarations that native or persona grading passed.

Rejected actions execute nothing. DeepSWE's protocol repair keeps the same schema, output budget and deadline, and identifies its feedback as host transport feedback rather than a user request. A DeepSWE shell action whose content holds a NUL character (U+0000) is rejected the same way, because no command line can carry it; this runtime change requires fresh calibration for new scored attempts. Scope and command-approval controls remain enforced by the existing controllers. Prompt changes require a fresh runtime fingerprint and calibration for new scored attempts.

## 3. Check the dataset before making API calls

```bash
swe-persona validate --dry-run --output runs/preflight
```

This checks all 100 rows, their content and assets, and the profile reconstruction/import paths. It starts no containers and makes no API calls. It does not establish that remote images are available or that a model can finish every task.

For a SWE-Live reference stability check, run:

```bash
python scripts/check_live_native.py --task swelive/0262-go--laptop_data_scientist \
  --repetitions 3 --output runs/live-native-check
```

Omit `--task` to check all 19 SWE-Live pairs. This starts fresh CPU containers and runs the unchanged official reference patch and tests, without model calls. It retains every repetition; any failure makes the check fail, even if later repetitions pass. It is a diagnostic release gate, not a task score or permission to retry candidate results until they pass. A reference failure must be investigated before claiming that the affected pair is reproducible. Scored SWE-Live attempts write `preflight-error.json` with the failed test IDs and `episode_started: false` when the reference fails.

Inspect task metadata without reading binary assets:

```python
import pyarrow.parquet as pq

tasks = pq.read_table("swe-persona-lite.parquet", columns=[
    "task_id", "source_benchmark", "interaction", "environment", "budget"
]).to_pylist()
for task in tasks:
    print(task["task_id"], task["interaction"]["profile_id"],
          task["environment"]["agent"]["resources"], task["budget"])
```

## 4. Run the agent and user simulator together

This CPU task is present in Lite:

```bash
swe-persona run \
  --task swepro/swepro-r0035-be662e3c3c-analyst \
  --config configs/api.local.json \
  --output runs/analyst-001
```

Use a new or empty output directory for each attempt. **This command starts the complete interaction and grading workflow**, including calls to the configured `user_simulator` API. You do not need to start a separate simulator process or construct persona prompts yourself.

The host controller:

1. Reconstructs the selected row and verifies its assets. It runs or reuses the required calibration for this runtime profile and API configuration.
2. Uses Harbor and Docker to prepare the coding environment, then gives the agent the profile's public opening and tools.
3. Sends observations and the allowed interaction context to the `user_simulator` API. The controller enforces the frozen disclosure schedule and returns the resulting user messages to the agent.
4. Repeats agent actions, user replies and profile-specific online checks until completion or a limit is reached. The coding model proposes actions; the harness executes them in the task environment.
5. Collects the workspace and runs the protected native tests and persona grading. The normalized result and detailed evidence are written to the output directory.

For TB3 `risk-scorer-replay`, private persona grading restores the shipped `/app/rebuild_parity_report.sh` to mode `0555` after copying the frozen artifacts, matching the native verifier's setup. Restoration requires the original regular file and checksum; missing, replaced or symlinked files remain for the existing checker to assess. The private copy's permission changes are recorded in `persona-check.json` under `checker_permission_restorations`; source contents and host artifacts are preserved. This runtime change requires fresh calibration for new scored attempts, with historical results retained separately.

Its persona checker also stops observing a CLI call when a manifest-selected source is not a regular file, such as a directory in place of a CSV, as it already does for an unreadable thresholds file, instead of recording the resulting `IsADirectoryError` as incomplete observation. This corrects the task row, so its `content_sha256` changes.

### Agent harnesses

The selected row's `interaction.profile_id` chooses the scaffold automatically. Public profile names use benchmark names; SWE-bench Pro adds `mini` or `shell` to distinguish its two agent scaffolds. The `source_subset` field uses the same names:

| Profile | Coding and interaction scaffold |
| --- | --- |
| `terminal-bench-3` | Harbor terminal agent with persona requirement progression. |
| `swe-bench-pro-mini` | Host mini-swe-agent with a Harbor coding container and persona controller. |
| `swe-bench-pro-shell` | ShellAgent with event-triggered user requirements and rubric checks. |
| `swe-bench-live` | Supervised shell interaction with the frozen handoff protocol and native preflight. |
| `swe-together` | API JSON shell agent with native follow-up disclosure and grounded persona checks. |
| `deep-swe` | API JSON action agent with the frozen staged interaction plan. |

### DeepSWE public evidence and simulator recovery

For a disclosed `grounded_explanation` requirement, the host records the registered public probes before rendering the next user message. An explanation required before implementation receives baseline observations only; a candidate explanation receives baseline and committed-candidate observations once implementation is authorized. Records are cached by probe ID and candidate patch digest; changed patches get new records. This measurement does not mark the requirement met or start final-quality acceptance early. Future, undisclosed probes are not run by this path. A failed public case is still delivered as an observed failure and may support an honest explanation.

Completion uses the structured action `{"action":"reply","phase":"done"}`. Completion prose in a working-phase reply does not emit that event. Simulator and review contexts distinguish these facts; requests for the event must not falsely assert that the agent never said it was done.

The simulator first renders ordinary prose and explicitly selects every current public document. One internal prose repair is allowed. If prose review still rejects the draft or returns unknown, one final simulator call selects the same documents with empty opening and closing prose. The host resolves only valid model-selected document IDs to their unchanged original bodies. This mode cannot add instructions, alter check statuses, authorize a transition or approve a rejected draft. Missing or future document IDs, malformed schema, receipt mismatches and cancellation still fail validation. No extra coding-agent turn is consumed by these internal repairs. Successful recovery is recorded in `simulator_message_recoveries`; native and persona final grading remain mandatory. These changes require a fresh runtime fingerprint and calibration for new scored attempts.

The two SWE-bench Pro implementations are packaged together under `src/swe_persona/vendor/swe_pro/`: `mini/` serves `swe-bench-pro-mini` and `shell/` serves `swe-bench-pro-shell`. The task profile selects the appropriate implementation automatically.

The shell adapter exports the frozen binding with `applicability.verdict: "applicable"`. Both dry runs and scored episodes use `registry.build_candidates` and `applicable_candidates`; preparation checks that all Parquet requirement IDs are loaded. Missing or contradictory applicability cannot silently turn a binding into an empty requirement list. The controller reads the shared passed calibration gate and validates its grading identity before registering these requirements in the ledger. Original event triggers, disclosure limits and optional/required schedule flags remain binding.

A progressive shell episode cannot earn joint credit with no persona checks. Final grading must cover every required or activated requirement. Missing loading or missing final checks is reported separately from a demonstrated persona violation; an unavailable persona grade is `null`. Result normalization compares the episode's loaded and graded IDs against the immutable row and disclosure record, so stale or inconsistent `joint_success` fields cannot manufacture a pass. Valid preserved work after an API interruption still follows the common recovery policy. These runtime changes require fresh calibration and new attempts; old episodes without persona disclosure cannot be repaired by post hoc grading.

These profiles share the public API configuration and command-line interface; their action protocols, schedules and graders remain source-specific. Harbor manages coding environments, and adapters also use Docker directly for snapshots and verification. To evaluate another **model**, change `agent` in the JSON configuration. This distribution does not expose a universal `--agent-class` plug-in or a standalone `/user-simulator` HTTP service. Replacing the scaffold itself requires adapting the profile code and constitutes a different evaluation setting.

The Python API clients can be used to check connectivity independently:

```python
from swe_persona.api import RoleAPI, load_config

config = load_config("configs/api.local.json", require_keys=True)
for role in ("agent", "user_simulator"):
    reply = RoleAPI(config, role).raw([
        {"role": "user", "content": "Reply with ready."}
    ])
    print(role, reply["text"])
```

This snippet makes two real API calls. It only checks the clients; it does not load a task, apply a persona or produce benchmark scores. Use `swe-persona run` for scored agent–user interaction.

### Calibration and information boundaries

Calibration runs automatically before a scored episode. It may make model calls and start containers. A blocked calibration produces `calibration_error` and prevents episodes for that profile. Other profiles in the batch continue. Gates are cached by profile, API configuration and runtime fingerprint under `~/.cache/swe-persona`. Use `--cache /path/to/cache` on `run` or the batch script to choose another location. SWE-bench-Live has a restricted fallback for unavailable calibration assessments. SWE-Together has a bounded diagnostic mode for reviewed positive-control disagreements, described below. A degraded gate is never reported as a passed calibration.

SWE-bench Pro shell uses `literal-clause-truth-v1` for rubric judge responses. Each clause returns `condition_holds: true`, `false`, or `null`, plus its reason and registered evidence IDs. The answer describes whether the **literal clause condition** holds; it is not a pass/fail verdict on the agent. For a `not_met_if` clause such as "an unauthorized edit occurred", `true` establishes that violation and `false` establishes its absence. For an `unknown_if` clause such as "the baseline is missing", a supplied baseline supports `false`. `null` means the eligible evidence cannot establish either truth value; it never means the condition is absent. Both `true` and `false` require eligible grounding. The host converts these answers to the existing clause aggregation vocabulary and retains the original rubrics, majority rule, deterministic checks and final native/persona scoring.

The shell calibration must distinguish both known correct and incorrect scope/explanation controls. Collection or judge errors cannot count as successful negative controls. Its report includes per-clause truth, votes and reasons so a contradictory assessment can be investigated. A failed gate blocks the profile's 17 pairs before agent execution; those entries are unresolved evaluations, not 17 demonstrated agent failures. The clause protocol is included in grading identity and cache keys; changed protocols require new calibration and separate scored attempts. Never bypass a failed gate or rewrite old results after changing the judge protocol.

The default shell calibration control is packaged in `runner/fixtures/rubric-calibration.json`: the unchanged pinned Navidrome task, reference patch and official runner/parser used by the existing controls. The loader verifies its dataset/harness revision, task field hashes and script hashes. It does not load Hugging Face data or download checker scripts; Docker may still need to pull the declared task image. Reference controls stay on the host and in isolated calibration containers, outside evaluated coding-agent environments. The calibration report records the fixture identity. An explicit upstream `--dataset-index` override in the vendored calibration tool is a separate mode and may need network access. A passed calibration and 17 successful dry runs establish protocol/setup readiness; publishing an end-to-end reproducibility claim still requires valid scored attempts for all 17 pairs.

The Harbor adapters used by Pro shell, SWE-Live and DeepSWE check the stopped session's default network after normal teardown and failed startup. They remove an empty network only when its Compose project label matches that session, using the inspected network ID. Live endpoints or a different owner prevent removal. This avoids accumulating unused networks until Docker's address pool is exhausted; they never perform a global prune. Each session writes `resource-cleanup.json`, including bounded cleanup errors when reclamation cannot finish. DeepSWE stores its host-owned session directories under the worker's `environments/` directory, with each `ds-*` Compose project derived from that directory's random session ID.

Pro shell names each episode container and its snapshot tags with a hash of the attempt's run directory, so batches that run the same task concurrently on one Docker daemon use distinct names and never remove each other's snapshots. Its episode and verifier containers are pinned to as many CPUs as their `--cpus` quota (`--cpuset-cpus`), so tools that size themselves from the visible CPUs, such as `pytest -n auto` and `nproc`, do not start one worker per host CPU inside the memory limit. Terminal-Bench 3 keeps the main runtime image for the post-teardown persona checker under two tags, because Compose's `down --rmi local --remove-orphans` removes the first sorted tag of each project image together with `<project>-main`.

After a worker exits, including watchdog timeout or interruption, the outer CLI also checks the known host-controlled bridge directories for empty session networks. It does not scan agent artifacts or follow session-directory symlinks. This resource reclamation has a separate ten-second bound outside the scored worker clock and writes `session-network-cleanup.json`; it does not change timeout outcomes or replace missing verification results.

DeepSWE online judges select host-registered references that must pass the same quote-provenance check used after resolution. Punctuation-only lines are not registered as evidence. This establishes citation validity only, not whether a requirement is fulfilled. A final judge response rejected after bounded repair cannot establish a persona pass: the native result remains separately available, while the persona result stays unknown and the joint attempt is classified as an evaluator infrastructure error.

API clients run on the host. Simulator and judge calls share the `user_simulator` connection settings while retaining separate role contexts. The controller supplies each role's permitted view; the agent is not given the entire Parquet row, hidden requirements, reference controls or protected tests. Some calibration fixtures and the SWE-Live native preflight contain host-only reference material required by their protocols. Do not mount the distribution or host cache wholesale into the coding container. The evaluator can inspect all distributed assets: this is an open benchmark, not a hidden-test service.

## 5. Evaluate a batch

Start with a small CPU batch:

```bash
python scripts/evaluate.py \
  --config configs/api.local.json \
  --output runs/cpu-smoke --cpu-only --limit 3
```

Evaluate all tasks that require no GPU:

```bash
python scripts/evaluate.py \
  --config configs/api.local.json \
  --output runs/lite-cpu --cpu-only
```

This selects **99 tasks**. The remaining task, `tb3/exam-pdf-eval`, declares one H100 GPU. To include all 100 tasks, first provide the required GPU and NVIDIA Container Toolkit, then explicitly select one device and omit `--cpu-only`:

```bash
export SWE_PERSONA_GPU_DEVICES=0
python scripts/evaluate.py \
  --config configs/api.local.json \
  --output runs/lite-all
```

The variable accepts one host GPU index or UUID; choose a device allocated to you. CPU tasks receive no GPU reservation. Different GPU hardware must be reported as an environment variant.

Optional batch selectors are `--source-benchmark "SWE-bench Pro"`, `--profile swe-bench-pro-shell` and `--limit N`. `--dataset` defaults to Lite. The driver runs up to two task workers concurrently by default; use `--workers N` to change this or `--workers 1` for sequential evaluation. Within a batch, each runtime profile is calibrated once before its tasks start, with calibration evidence shared through the cache. Each task uses its own numbered output directory and driver log. `summary.json` is updated atomically after task completions, preserves dataset order, and includes active tasks and worker count. A failed calibration blocks only its own profile: affected rows receive `calibration_error` with `episode_started: false`, `blocked_profiles` identifies the affected profiles, and other profiles continue. SIGINT/SIGTERM stops launching tasks and interrupts active task processes for cleanup. Invalid/error outcomes are reported separately. The command returns exit code 0 only when every selected task has a valid pass/fail result (or a successful dry run); a valid model failure does not make the command itself fail. Repeated attempts must use separate directories and must not be combined by selecting the best score.

## 6. Time budgets

All profiles expose the same agent/user interaction time override:

```bash
swe-persona run \
  --task swepro/swepro-r0035-be662e3c3c-analyst \
  --config configs/api.local.json --output runs/analyst-custom-budget \
  --timeout-sec 3600

python scripts/evaluate.py \
  --config configs/api.local.json --output runs/custom-budget \
  --cpu-only --total-timeout-sec 3600
```

`--timeout-sec` and `--total-timeout-sec` are aliases and require a positive finite value. Omit them to retain each row's `budget.episode_timeout_sec`:

| Source | Lite default interaction limits |
| --- | --- |
| Terminal-Bench 3.0 | 2,500–28,800 seconds |
| SWE-bench Pro | 3,600 seconds |
| SWE-bench-Live MultiLang | Approximately 156–213 seconds |
| SWE-Together | 1,800 seconds |
| DeepSWE | 3,600 seconds |

The common `agent-user-wall-clock-v2` budget starts when the task environment and preflight are ready for formal interaction. It covers agent and simulated-user calls, online judges, tools, and interaction snapshots. Worker startup, image download/build, container/environment preparation, native reference preflight, prior calibration, input materialization, and final independent verification do **not** consume this budget. The legacy `--total-timeout-sec` alias also sets the interaction cap, not a preparation-plus-interaction cap. Existing profile-specific coding, turn, action, and native verification limits still apply; increasing the override does not disable them.

Host controllers publish one interaction deadline after preparation; watchdogs and nested SWE-Live workers use this deadline instead of restarting the same duration at different stages. Preparation has a separate 7,200-second infrastructure watchdog. SWE-Live's official gold preflight, which applies the reference patch and runs the native tests before interaction starts, has its own wall of at least 300 seconds. A larger `verifier_timeout_sec` is preserved. Final verification has its own watchdog, defaulting to `max(900, budget.verifier_timeout_sec + 600)` seconds; the original native verifier's own limit remains unchanged. Set `SWE_PERSONA_PREPARATION_TIMEOUT_SEC` or `SWE_PERSONA_VERIFICATION_TIMEOUT_SEC` to positive finite values to configure these infrastructure limits. `time-limit.json` records phase events and separate preparation, interaction, and verification elapsed times. A bounded transition/cleanup grace grants no extra interaction credit.

An interaction timeout is recorded as `status: "fail"`, `outcome_reason: "total_timeout"` and zero joint credit. Available independent native/persona grades are retained; missing grades remain unknown. A preparation or final-verification watchdog timeout is `infrastructure_error`, with `preparation_timeout` or `verification_timeout`, rather than a coding failure. This policy changes the effective interaction allowance relative to `worker-wall-clock-v1`; use fresh calibration and new output directories, retain historical attempts, and compare runs under the same policy and overrides.

### API timeout recovery across all subsets

TLS disconnects (`SSLEOFError`, `SSLZeroReturnError`, or `SSLError` with reason `UNEXPECTED_EOF_WHILE_READING`) use the same bounded transport recovery, including when wrapped by urllib or an explicit exception cause. Certificate-verification failures and unrecognized SSL errors fail without retry. Certificate verification remains enabled. This runtime change requires fresh calibration and separate scored attempts; historical `SSLError` records that omitted the underlying reason cannot establish whether those failures would recover.

Every executable profile uses `api-recovery-and-artifact-assessment-v1`: Terminal-Bench 3.0, SWE-bench Pro mini/shell, SWE-bench-Live, SWE-Together, and DeepSWE. A transient API timeout, connection error, rate limit, or server error is retried up to three total HTTP attempts, with 1- and 2-second backoff. The configured API timeout limits each HTTP attempt; retries and backoff consume the existing interaction/sample budget. SDK retries and outer transport retries do not multiply this limit. A cancelled task, authentication failure, or schema violation is not treated as a transient API timeout. JSON format repair remains a separate bounded operation.

If transport recovery is exhausted, the controller preserves the current workspace and transcript and attempts its independent final native/persona verification within the remaining budget. It does not replay shell commands, restart the task, invent user approvals, or substitute online judge results for missing final grades. Preserved work can be a patch, a task artifact, a substantive agent message, or recorded completed agent tool work, depending on the profile. A run with no usable work or no final artifacts is `infrastructure_error`. A run with preserved work passes only when independent native and persona verification both pass; an unmet or unverified required component receives `fail` and zero joint credit. Missing component grades remain `null`, and `verification_complete` records whether both components were graded. Artifact integrity violations remain `invalid`. Total task/interaction deadlines retain precedence and score as failures.

Recovery attempts are audited in `transport-recovery.jsonl` (DeepSWE also records recovery events in its API JSONL). Interrupted results include `outcome_reason: "api_interrupted"` and `api_recovery_assessment`, including available artifacts, agent progress, observed component grades, and the assessment policy. This changes the runtime fingerprint, invalidates earlier calibration caches, and requires fresh calibration for subsequent real evaluation. Results produced under the earlier policy should remain separate.

### Recoverable online requirement assessments

Online assessments use `registered-refs-internal-recovery-v2`. In Terminal-Bench 3.0, the judge returns criterion statuses and `evidence_refs`: registered agent-message IDs after the requirement disclosure. The response schema enumerates eligible IDs and frozen requirement/criterion IDs. The judge never copies or generates evidence quotations. The controller retrieves the original ledger text unchanged and records it in `requirement-judge-audit/review-evidence.jsonl`. Local validation still checks criterion coverage, valid evidence IDs, nonempty evidence for `MET`, verdict aggregation, and actionable feedback.

A semantically invalid assessment receives at most two total assessment attempts against the same evidence and rubric, with controller validation feedback on the second attempt. Both share the original assessment deadline and consume the existing sample budget. Transport recovery and JSON format repair retain their own bounded limits; exhausted transport calls are not retried again by the assessment layer. Rejections are audited in `requirement-judge-audit/review-recovery.jsonl`.

An internal judge failure is **assessment unavailable**, separate from a valid `UNOBSERVABLE` verdict. It preserves the last accepted verdict, blocks acceptance and dependent requirement disclosure, and does not call the user simulator, add a corrective user message, or consume a user turn solely to fix the judge. At a progress checkpoint the agent continues working. At completion, persistent assessment failure ends interaction with `online_judge_unavailable`, preserves the current artifacts, and runs independent final native/persona verification within the remaining budget. Only complete passing verification earns joint credit; unresolved required grades remain unknown and receive zero joint credit when usable work is present. A previous online `MET` alone cannot establish final success. Genuine validated unmet or unobservable requirements still receive normal user feedback.

SWE-Together also retries invalid assessments internally at most twice, retains accepted verdicts in `last_valid_verdicts`, and records unavailable assessments as `unknown` with separate `judge_errors`. It does not turn internal judge errors into requests for repeated explanations or permanent invalid flags. A completion assessment that remains unavailable stops interaction for final verification. Later valid assessments can restore readiness; frozen bindings, evidence integrity, simulator policy checks, and total deadlines retain their existing requirements. These protocol changes require fresh calibration and a separate evaluation batch.

### SWE-Together command timeouts and silent interaction stalls

Harbor command timeouts are returned to the coding agent as a tool result with exit code `124` and `timed_out: true`. The per-command cap remains 180 seconds, bounded by the remaining coding budget. Other runtime errors propagate. A host timeout can leave the container command running; the agent is told to inspect process state before retrying.

The `silent-no-progress-v2` policy ends interaction after four consecutive simulator `no-op`/`wait` decisions with the same workspace snapshot and no intervening agent commands. Any command, workspace change, or simulator message resets the count. Counting is suspended while any frozen persona requirement or mandatory native follow-up remains undisclosed, so an early cutoff cannot prevent its scheduled release. This detector measures silent message-only loops; repeated user messages and changing explanations are not classified semantically. Existing task and step limits still apply.

Set `SWE_PERSONA_TOGETHER_STALL_TURNS` to a nonnegative integer to change the threshold; `0` disables this detector. The default is `4`. Both the episode and normalized result record the effective `interaction_stall_policy`; use the same threshold when comparing models and report overrides alongside scores.

A stall sets `stop_reason: interaction_stalled`, preserves the transcript, command history and persona evidence, and runs the existing final persona review and native verifier. `interaction_stall_assessment` records the observed component grades and verification coverage. Preserved work with an unmet or unverified required component receives `fail` with zero joint credit; unknown grades remain `null`. Evidence integrity violations remain `invalid`, and a native verifier that cannot run remains an infrastructure error. Silence never supplies user acceptance or authorization. The cutoff itself neither grants a pass nor skips verification. `stall-evidence.json` records the triggering turns, workspace snapshot and threshold. This runtime change invalidates earlier calibration caches; subsequent evaluations require fresh calibration and must be reported separately from earlier attempts.

SWE-Together calibration uses 37 fixed semantic controls: 18 positive, 18 negative, and one missing-evidence control expected to return `unknown`. The `swe-together-calibration-fixtures-v3` suite declares every control ID, expected label, binding identity, required artifact, source checksum, and execution receipt. Missing controls, missing provenance, changed source artifacts, and incomplete command inventories fail validation before model calls. The 26 packaged source files include the full pre-change CLI entrypoint and exact old/new Windows anonymizer sources. Windows walkthroughs state the actual hashed-prefix and backslash outputs, including the unchanged slash-only subdirectory condition. Native patches, persona requirements, and the Parquet dataset are unchanged.

PR handoff inventories cover **every** declared package script and frozen verifier entry. Operational scripts such as deployment, cleanup, content generation, and service startup are listed as not run; listing grants no permission to execute them. Patch-only handoffs claim no execution. The learning-constraint fixture instead contains actual bounded static-check receipts replayed on pinned files plus the reference patch: shell syntax, diff whitespace, and the packaged verifier's structural acceptance patterns. These receipts describe a calibration fixture, not an agent episode or Docker image/E2E run. `fixture-validation.json` reports six receipt entries across its positive/negative pair. Other narrow source excerpts remain explicitly bounded synthetic controls; they do not certify complete repository behavior. Negative controls use distinct task-specific violations rather than one repeated multi-violation answer. A judge error, invalid response, or API interruption cannot count as a successful `unknown` control.

#### SWE-Together startup policy

The default `SWE_PERSONA_TOGETHER_CALIBRATION_MODE=bounded` permits diagnostic execution after **at most two** semantic disagreements on the six reviewed positive controls declared in `calibration-suite.json`: agent-swarm handoff and learning constraints; amytis handoff; CLI old-path explanation; and the two Dataclaw Windows explanations. The remaining positives, all 18 negatives, the missing-evidence control, and all 18 simulator disclosure controls must pass. Any assessor/API error, incomplete report, or failed source/fixture validation blocks startup. This is a fixed tolerance policy, not a claim that 35/37 establishes general judge accuracy. Calibration is sampled once; failures are not retried to select a favorable semantic verdict.

An allowed disagreement produces `status: "degraded"`, `passed: false`, `execution_allowed: true`, and `score_provisional: true`. The full audit and disagreement IDs remain in the calibration report; the unified gate, run configuration, and task result preserve this distinction, including timed-out attempts with no episode file. Native and persona verification still run. The score summary retains observed component grades and reports provisional joint rates, while formal `joint_success_rate` and `joint_success_rate_valid_only` remain `null` for groups containing provisional attempts. Use `SWE_PERSONA_TOGETHER_CALIBRATION_MODE=strict` for runs requiring every control to match. The mode is part of the calibration cache identity. Corrected fixtures and the new runtime fingerprint require fresh calibration and separate attempts; historical results are not rewritten.

### Runtime recovery and native failure classification

SWE-Together sends the actual simulator decision schema to the provider. Strict schema mode declares all six fields as required; `base_task_complete` and `completion_reason` are `null` for ordinary messages and silence. Acceptance still requires `true`, a nonempty completion reason, and the independent disclosure/persona gates. Rubric and interaction audit roles each have their own complete schema. Calibration also exercises 18 actual simulator disclosure calls, one per independent control binding (including all four retained Lite pairs), before agents start; all must pass without host fallback.

Simulator format/policy repair is bounded within the controller and shares the task deadline. An exhausted reply can recover an already due request using its frozen statement and host schedule. This recovery is recorded in `simulator_errors` and does not leave a permanent invalid flag. Other exhausted replies stop interaction for final native/persona verification rather than requesting repeated agent explanations. Recovery never invents authorization or acceptance; missing disclosures or unresolved assessments cannot earn joint credit.

Shell command output uses explicit UTF-8 decoding with replacement for malformed bytes on normal completion and timeout. Checker verdict lines are split at newline characters only, so evidence that quotes UTF-8-decoded binary files (for example Redis dump files left in the repository) keeps its verdict instead of becoming `driver_no_verdict`. This runtime change requires fresh calibration and separate scored attempts. SWE-Live inventories official test paths in an isolated working directory so an upstream checkout subdirectory cannot silently make the inventory empty. Candidate changes to those test files are excluded from the evaluated patch and recorded in the projection receipt; the original submission and official test patch remain unchanged.

When a shell native run observes no required tests, a compiler diagnostic alone is insufficient to declare functional failure. The evaluator runs the frozen base plus reference patch and the same official tests in an isolated worktree. The 17 shell reference controls are packaged separately in `native-reference-controls.json`, pinned to the original dataset revision and checked against each row’s base commit, test patch, runner and parser hashes. They are mounted only for native diagnosis; ordinary agent and public checker mounts never include them. Executable checker logic comes from the current runtime while test scripts and data remain frozen. Only a control passing every required test allows `official_candidate_compilation_failed` with `failure_category: "candidate_code"`. If the control cannot run or pass, or the output instead indicates a runner/dependency/resource fault, native status remains `unknown` with `failure_category: "test_environment_unavailable"`. Reference control details remain evaluator-only. Historical attempts are preserved; these changes require fresh calibration and new scored attempts.

### SWE-bench-Live calibration and judge fallback

The SWE-Live runtime uses `swe-live-tb3-handoff-loop-v3`; the dataset's original handoff requirements remain unchanged and their source protocol is recorded separately. Calibration and online judgments use the same registered `evidence_refs` schema, eligible message IDs, prompts and validation. Up to two assessment attempts share one bounded deadline and identical evidence. Exhausted transport requests are not retried again by this layer. Original message text is retrieved by the controller and saved unchanged in the judge audit.

All 12 calibration controls are processed even when an assessment remains unavailable. Such controls are recorded as `status: "unknown"` with `assessment_unavailable: true` and error diagnostics; they do not pass. A validated verdict outside a control's expected verdicts is a semantic failure and blocks this profile. When there are unavailable controls but no semantic failures, calibration is explicitly `status: "degraded"`, `passed: false`, `execution_allowed: true`, with `independent_checker_required: true`. Incomplete calibration reports cannot authorize execution. The unified gate, run configuration and calibration report preserve this distinction.

This fallback applies only to the complete literal handoff contract implemented by SWE-Live's frozen deterministic checker: exact labeled lines, nonempty bullet bodies, the word limit, and the manager's code-marker restrictions. Materialization verifies that the checker covers the entire frozen disclosure and rubric before permitting execution. It cannot grade arbitrary persona requirements or repair correctness. If an online model assessment remains unavailable, its audit retains `unknown`; a separate `independent_frozen_handoff_checker` assessment can resolve this literal contract and support ordinary interaction without asking the agent to repair judge output. Its diagnostic `assessment_recovery` is retained in the episode. Genuine handoff violations still receive corrective feedback. Final independent native verification and persona checking remain mandatory; unknown calibration alone earns no task credit. Cancellation, artifact integrity failures and sample deadlines retain precedence.

## 7. Read complete scores and evidence

Every scored attempt writes:

| File | Contents |
| --- | --- |
| `run-config.json` | Task/content identity, runtime/API fingerprints, model settings and effective time policy. |
| `result.json` | Normalized outcome, native/persona grades and pointers to detailed results. |
| `time-limit.json` | Default and effective limits, measured duration and timeout/cleanup information, when the worker ran. |
| `run.log` and profile subdirectories | Execution diagnostics, interaction records and verifier evidence. |

The paths in `result.json.details` are relative to that attempt's output directory. Depending on the profile, they point to an `episode.json` or `persona-result.json`; these contain the source-specific requirement assessments and references to further evidence. API/calibration failures can prevent some files or grades from being produced. Output logs contain newly generated task context and interactions; keep them outside the dataset distribution.

The normalized grading fields are:

| Field | Interpretation |
| --- | --- |
| `native_pass` | Original task's test outcome: `true`, `false`, or `null` when unavailable. |
| `persona_pass` | Persona requirement outcome: `true`, `false`, or `null` when unavailable. |
| `joint_success` | Whether the profile reports a valid joint pass. Read it with `status`; errors also carry `false`. |
| `status` | `pass` / `fail` for valid outcomes; `invalid`, `infrastructure_error` or `calibration_error` for incomplete/invalid evaluation. |
| `outcome_reason` | Additional cause when provided, including `total_timeout`. |

Full component grading means a valid pass/fail outcome with both `native_pass` and `persona_pass` present as booleans, and verification not marked incomplete. A model may fail while the evaluation completes successfully. A budget timeout receives zero joint credit even if final component grading was not reached. `null` does not mean a failed native test or a successful persona check.

Summarize either a batch or one task without making further API calls:

```bash
python scripts/score.py runs/lite-cpu/summary.json \
  --output runs/lite-cpu/scores.json

python scripts/score.py runs/analyst-001/result.json
```

The score report contains overall, per-source and per-profile metrics. Rates are fractions in `[0, 1]`:

- `joint_success_rate`: joint passes divided by the full selected count, emitted only when every selected task has a valid pass/fail outcome and none is provisional. Otherwise it is `null`.
- `joint_success_rate_valid_only`: joint passes divided by valid outcomes, with `valid_evaluations` as its denominator. It is a partial-coverage diagnostic when errors remain, and is `null` if any attempt in the group is provisional.
- `score_status`, `provisional_evaluations`, and `provisional_joint_success_rate[_valid_only]`: identify diagnostic attempts allowed by degraded calibration and report their separately labeled rates. Observed native/persona rates retain their displayed denominators.
- `native_pass_rate_observed` and `persona_pass_rate_observed`: rates over available component grades from valid outcomes, with their observed counts reported separately.
- `complete_native_and_persona_grading`, `total_timeouts`, `status_counts` and `not_attempted`: coverage and completion information that must accompany rates.

Do not present a valid-only rate as the score for all 100 tasks. Report the exact selection, runtime/API fingerprints, shared user-simulator/grading settings, time limits and hardware alongside scores. Profile results should be shown separately because the scaffolds differ.

## Data format and release scope

One Parquet row contains one frozen task/persona pair. `source_benchmark` identifies its upstream benchmark. The remaining columns encode source identity, the opening, public agent context, native task information, persona, requirements, interaction profile, environment, budgets, evaluation rules, artifacts, embedded assets and a content digest. The executable contract is `src/swe_persona/schema.py`.

The materializer reconstructs hash-checked build contexts, tests and controller bindings. Images and external downloads remain runtime dependencies. Source trajectories and historical model scores are not required for evaluation. The Lite sample is a fixed proportional sample of an adapted cohort, not a claim of an unbiased sample of all upstream tasks; the DeepSWE source cohort was retrospectively curated.

This release does **not** claim that all 100 tasks have completed successful environment and end-to-end validation. Image support, valid structured responses, available external images/assets, and adequate task-specific resources are required. The SWE-Live `swelive/0262-go--laptop_data_scientist` reference has produced an intermittent failure in `Test_handler_download_logs`; a later successful check does not erase that observation or certify stability. Historical model runs are not bundled or silently resumed by this distribution. A new `run` command starts a new attempt and its own required calibration. No task has been removed or replaced to improve model results.

Before publishing a reproducibility claim, verify the release checksums, install the built wheel in a clean environment, complete all 100 dry runs, inspect reference stability reports, and obtain valid scored attempts for every included pair with the documented simulator configuration. Calibration and infrastructure errors are unresolved evaluations, not capability failures. Run the local regression suite with `python -m unittest discover -s tests -v`; it makes no external model calls or real container launches.

Harness code is MIT-licensed. Upstream code, tests and assets retain their own licenses. Preserve the included notices under `src/swe_persona/vendor/` and the task notices identified by `source.license_asset_ids`; the harness license does not relicense upstream material.
