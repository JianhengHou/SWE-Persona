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

Use Linux, Python 3.12, Docker Engine with the Compose plugin, and permission to run Docker. On macOS, use Docker Desktop 4.86.0 or newer. The harness pins Harbor 0.23.0 and mini-swe-agent 2.4.6. Image downloads and builds require network access and sufficient disk space. Task CPU, memory, storage and GPU requirements are recorded in Parquet.

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

| Configuration section | Responsibility |
| --- | --- |
| `agent` | The coding model being evaluated; produces actions and replies to the user. |
| `user_simulator` | Provides persona interaction and evidence-based persona grading through the same API. Native tests run separately in Docker. |

```bash
cp configs/api.example.json configs/api.local.json
export SWE_PERSONA_AGENT_API_KEY='YOUR_AGENT_API_KEY'
export SWE_PERSONA_SIMULATOR_API_KEY='YOUR_SIMULATOR_API_KEY'
```

Edit **both sections** of `configs/api.local.json`. Replace each `base_url` and `model` placeholder with your provider's endpoint and model identifier. `base_url` is the API root, such as `https://provider.example/v1`, without `/chat/completions`. `api_key_env` names an environment variable; it does not contain the key. Do not put credentials in JSON or Parquet. The two variables may hold the same key when one provider serves both connections. No separate judge key or endpoint is required.

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

The full JSON must contain only `agent` and `user_simulator`. Both use Chat Completions in the generic example. The adapter also supports `api: "responses"`; the `swe-bench-pro-mini` coding agent specifically requires Chat Completions. Keep `user_simulator` fixed when comparing coding models; change only `agent`.

Interaction and grading remain separate model calls with separate prompts, allowed evidence and audit records. The simulator's approval is not itself a passing score. The loader creates an internal `judge` alias from `user_simulator` for the profile adapters; this alias may appear in `run-config.json` and logs, but is not a third API configuration. Adding a separate `judge` section to the input configuration is rejected.

Provider options:

- `structured_output_mode`: `json_schema` requests schema-constrained output; `json_object` requests JSON mode; `prompt` supplies the schema in instructions without sending a provider `response_format`. Local validation still applies. Use `prompt` for providers that cannot enforce the schema. Setting only `strict_json_schema: false` selects `json_object`, not `prompt`.
- `max_tokens_field`: for Chat Completions, select `max_completion_tokens` or `max_tokens` according to the provider. A positive integer limit in `request_options` overrides the profile's per-call output limit. Responses uses `max_output_tokens` instead. Reasoning models may consume this output budget before producing a visible answer.
- `request_options`: provider-supported settings such as `temperature`, `reasoning_effort` and `extra_body`. Omit unsupported options. DeepSWE supports `reasoning_effort`, `temperature`, `top_p`, token limits, `seed`, `stop` and `extra_body`.
- `timeout_sec`: the API client's request timeout; some profiles supply their own per-call timeout. It is separate from the task's total time budget.

For the supplied GLM-5.3 calling convention, copy `configs/api.zai.example.json` instead. It selects prompt-based structured output and `max_tokens`. The tested GLM endpoint rejected image input; it is not a compatible coding endpoint for every Lite task. A text-only endpoint is not made multimodal by this harness.

## 3. Check the dataset before making API calls

```bash
swe-persona validate --dry-run --output runs/preflight
```

This checks all 100 rows, their content and assets, and the profile reconstruction/import paths. It starts no containers and makes no API calls. It does not establish that remote images are available or that a model can finish every task.

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

The two SWE-bench Pro implementations are packaged together under `src/swe_persona/vendor/swe_pro/`: `mini/` serves `swe-bench-pro-mini` and `shell/` serves `swe-bench-pro-shell`. The task profile selects the appropriate implementation automatically.

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

Calibration runs automatically before a scored episode. It may make model calls and start containers. A failed gate produces `calibration_error`; it must be resolved before the episode is scored. Successful gates are cached by profile, API configuration and runtime fingerprint under `~/.cache/swe-persona`. Use `--cache /path/to/cache` on `run` or the batch script to choose another location.

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

Optional batch selectors are `--source-benchmark "SWE-bench Pro"`, `--profile swe-bench-pro-shell` and `--limit N`. `--dataset` defaults to Lite. The driver runs sequentially, stores per-task outputs in numbered directories and updates `summary.json`. It stops on calibration failure and reports other invalid/error outcomes separately. It returns exit code 0 only when every selected task has a valid pass/fail result (or a successful dry run); a valid model failure does not make the command itself fail. Repeated attempts must use separate directories and must not be combined by selecting the best score.

## 6. Time budgets

All profiles expose the same total-time override:

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

| Source | Lite default total limits |
| --- | --- |
| Terminal-Bench 3.0 | 2,500–28,800 seconds |
| SWE-bench Pro | 3,600 seconds |
| SWE-bench-Live MultiLang | Approximately 156–213 seconds |
| SWE-Together | 1,800 seconds |
| DeepSWE | 3,600 seconds |

The common `worker-wall-clock-v1` limit includes worker startup, image/environment preparation, native preflight, agent and simulator work, judges, and final verification. Prior calibration and input materialization are outside this clock. Existing profile-specific coding, turn, action and verification limits still apply; increasing the outer cap does not disable them. Cleanup receives a bounded grace period outside the scoring budget.

A total timeout is recorded as `status: "fail"`, `outcome_reason: "total_timeout"` and zero joint credit. Missing native/persona grades remain unknown. Compare runs under the same budget policy and record overrides; do not silently extend short defaults to obtain a better score.

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

- `joint_success_rate`: joint passes divided by the full selected count, emitted only when every selected task has a valid pass/fail outcome. Otherwise it is `null`.
- `joint_success_rate_valid_only`: joint passes divided by valid outcomes, with `valid_evaluations` as its denominator. It is a partial-coverage diagnostic when errors remain.
- `native_pass_rate_observed` and `persona_pass_rate_observed`: rates over available component grades from valid outcomes, with their observed counts reported separately.
- `complete_native_and_persona_grading`, `total_timeouts`, `status_counts` and `not_attempted`: coverage and completion information that must accompany rates.

Do not present a valid-only rate as the score for all 100 tasks. Report the exact selection, runtime/API fingerprints, shared user-simulator/grading settings, time limits and hardware alongside scores. Profile results should be shown separately because the scaffolds differ.

## Data format and release scope

One Parquet row contains one frozen task/persona pair. `source_benchmark` identifies its upstream benchmark. The remaining columns encode source identity, the opening, public agent context, native task information, persona, requirements, interaction profile, environment, budgets, evaluation rules, artifacts, embedded assets and a content digest. The executable contract is `src/swe_persona/schema.py`.

The materializer reconstructs hash-checked build contexts, tests and controller bindings. Images and external downloads remain runtime dependencies. Source trajectories and historical model scores are not required for evaluation. The Lite sample is a fixed proportional sample of an adapted cohort, not a claim of an unbiased sample of all upstream tasks; the DeepSWE source cohort was retrospectively curated.

This release does **not** claim that all 100 tasks have completed end-to-end validation with GLM-5.3. Image support, valid structured responses, available external images/assets, and adequate task-specific resources are required. Previous model runs were stopped; they are not bundled or silently resumed by this distribution. A new `run` command starts a new attempt and its own required calibration. No task has been removed or replaced to improve model results.

Harness code is MIT-licensed. Upstream code, tests and assets retain their own licenses. Preserve the included notices under `src/swe_persona/vendor/` and the task notices identified by `source.license_asset_ids`; the harness license does not relicense upstream material.
