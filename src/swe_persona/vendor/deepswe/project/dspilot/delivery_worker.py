"""Subprocess worker for the portable release and its versioned staged engine."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('validate', 'preflight', 'run', 'export-results', 'rules', 'calibrate', 'environment'))
    p.add_argument('--pipeline-id'); p.add_argument('--arm', default='pivot')
    p.add_argument('--pairs-json'); p.add_argument('--output', type=Path)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--opening-directory', type=Path)
    p.add_argument('--calibration-dry-run', action='store_true')
    p.add_argument('--calibration-reuse', action='store_true')
    a = p.parse_args()
    # Harbor starts in this child. The parent patch does not cross the process boundary.
    from swe_persona.harbor_bridge import align_egress_sidecar_platform
    align_egress_sidecar_platform()
    from . import specs
    if a.operation == 'calibrate':
        from .delivery_calibration import dry_run, run
        value = dry_run() if a.calibration_dry_run else run(a.dataset, a.output, reuse=a.calibration_reuse)
    elif a.operation == 'environment':
        from .frozen_staged import environment_check
        value = environment_check(specs.compile_episode(os.environ['DS_TASK_ID'], os.environ['DS_PERSONA_ID']))
    elif a.operation in ('validate', 'rules'):
        from .staged import adapt_episode
        from .frozen_staged import basic_checks
        from .planning_staged import source_manifest
        from .prepare import verify_cache
        from .check_contracts import checker_route, planning_evidence_contract, probe_pass_required
        from .checks_staged import artifact_names
        from .requirement_judge import rubric as online_rubric
        def delivery_artifacts(req):
            names = list(artifact_names(req))
            checker = req['checker']; params = checker.get('parameters', {})
            # Match the existing defaults in checks.isolated_patch/handoff.
            if checker['kind'] == 'isolated_patch': names.append(params.get('manifest', 'scope.json'))
            if checker['kind'] == 'reviewable_handoff':
                names.extend([params.get('draft', 'PR.md'), params.get('manifest', 'validation.json')])
            return sorted(set(names))
        results = []
        for pair in json.loads(a.pairs_json):
            ep = specs.compile_episode(pair['task_id'], pair['persona_id'])
            plan = json.loads((specs.ROOT / 'data/staged_bindings' / f'{ep.task.id}--{ep.persona["id"]}.json').read_text())
            basic_checks(plan, ep); source_manifest(ep); verify_cache(ep.task.id)
            adapted = adapt_episode(ep, plan=plan)
            templates = {r['id']: r for r in ep.persona['attributes']['evolved_requirements']}
            rules = [{'requirement_id': r['id'], 'effective_requirement': r,
                      'online_rubric': online_rubric(r, ep.persona),
                      'satisfaction': templates[r['template']]['satisfaction'],
                      'lifecycle': plan['requirements'][r['id']],
                      'artifact_paths': delivery_artifacts(r),
                      'route': checker_route(r, plan['requirements'][r['id']]),
                      'evidence_contract': planning_evidence_contract(r, plan['requirements'][r['id']]),
                      'probe_pass_required': probe_pass_required(r, plan['requirements'][r['id']]),
                      'assessment_kind': 'programmatic' if checker_route(r, plan['requirements'][r['id']]) in ('early_restore', 'candidate_trusted_probe') else 'mixed_programmatic_and_llm'}
                     for r in adapted.requirements]
            results.append({'pair_id': pair['delivery_id'], 'passed': True,
                            **({'rules': rules, 'effective_binding': adapted.binding.raw} if a.operation == 'rules' else {})})
        value = results
    elif a.operation == 'export-results':
        from .export_tb3_results import export_batch
        value = export_batch(specs.ROOT, specs.ROOT / 'runs/delivery-batch', a.output, work_dir=specs.ROOT.parent)
    else:
        from .staged_pipeline import execute
        if a.operation == 'preflight':
            value = execute(a.pipeline_id, task_id=os.environ['DS_TASK_ID'], persona_id=os.environ['DS_PERSONA_ID'],
                            command='freeze', arm=a.arm, delivery_opening=a.opening_directory)
            if value['status'] != 'prepared': raise ValueError('Fresh freeze failed: ' + str(value.get('reasons')))
            from .frozen_staged import environment_check
            value['environment'] = environment_check(specs.compile_episode(os.environ['DS_TASK_ID'], os.environ['DS_PERSONA_ID']))
        else:
            value = execute(a.pipeline_id, command='run', arm=a.arm, prepared_only=True)
    print(json.dumps(value, ensure_ascii=False))
    if isinstance(value, dict) and a.operation == 'run' and value.get('status') != 'complete': raise SystemExit(1)


if __name__ == '__main__': main()
