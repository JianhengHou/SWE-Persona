"""Execute a frozen handoff requirement loop with independent native verification."""
import bootstrap_adapter
from bootstrap_adapter import HERE
import argparse
import json
from pathlib import Path
import uuid
from agent import TerminalAgent
from llm import Llm, CallLog
from support import Manifest, Runtime, Snapshots, Checkers, read, file_hash
from execution_policy import BudgetClient, Deadline
from run_reference import gold_proof
from roleplay import FIT,fit_messages,validate_fit
from build_task import build, validate, save
from protocol import PROTOCOL, JUDGE_SCHEMA, SIM_SCHEMA, SIMULATOR_ENVELOPE, validate_judgment, eligible, checker, benchmark_outcome, visible_persona


def judge(llm, req, persona, public, messages, turn, audit, prompt, rubric, final=False):
    payload={'persona':visible_persona(persona),'public_constraints':public,'rubric':rubric,
             'eligible_visible_messages':eligible(messages,2),'final_recheck':final}
    result=llm.json_obj([{'role':'system','content':prompt}, {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
        JUDGE_SCHEMA,schema_name='requirement_judge',role='requirement.final' if final else f'requirement.turn{turn}',max_output_tokens=1800)
    validate_judgment(result,req,messages,2)
    save(audit/f'{turn:02d}-{"final" if final else "checkpoint"}.json',
         {'input':payload,'output':result})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True); p.add_argument('--persona',required=True)
    p.add_argument('--gold-dir',type=Path,required=True); p.add_argument('--binding-dir',type=Path,required=True)
    p.add_argument('--model',default='gpt-6-astra'); p.add_argument('--sample-seconds',type=float,required=True)
    p.add_argument('--deadline-monotonic',type=float); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    validate(a.binding_dir)
    if a.dry_run:
        print(json.dumps({'task':str(a.binding_dir),'valid':True,'model_calls':0})); return 0
    run=a.output_dir.resolve(); run.mkdir(parents=True,exist_ok=False)
    private=run/'agent-private'; private.mkdir(); audit=private/'requirement-judge-audit'; audit.mkdir()
    (private/'audit').mkdir()
    task=Manifest(a.task); bound=a.binding_dir/'tests/persona'; persona=read(bound/'persona.json')
    req=read(bound/'evolved_requirements.json')['requirements'][0]
    if persona['id']!=a.persona or read(bound/'build-identity.json')['source_task_sha256']!=task.task_sha256:
        raise ValueError('Task/persona binding identity mismatch')
    deadline=Deadline(a.sample_seconds,'sample',a.deadline_monotonic)
    llm=BudgetClient(Llm(model=a.model,call_log=CallLog(path=private/'audit/llm_calls.jsonl')),90); llm.deadline=deadline
    public=task.public_view(); messages=[]; decisions=[]; runtime=None; met=False; satisfied=False; result=None
    judge_prompt=(bound/'closed_loop/judge_prompt.md').read_text(encoding='utf-8')
    simulator_prompt=(bound/'closed_loop/simulator_prompt.md').read_text(encoding='utf-8')
    rubric=read(bound/'closed_loop/rubric.json')
    save(run/'protocol.json',{'protocol':PROTOCOL,'model':a.model,'persona':a.persona,'task_sha256':task.task_sha256,
        'binding_sha256':file_hash(bound/'binding.json'),'sample_cap_seconds':a.sample_seconds,
        'max_user_turns':24,'max_simulator_decisions':64,'test_patch_policy':'v2 protected official-test-file projection'})
    save(run/'task.public.json',public); save(run/'persona.json',persona)
    def checkpoint():
        episode={'protocol':PROTOCOL,'task_id':task.task_id,'persona':a.persona,'transcript':messages,
            'requirements':[req], 'disclosures':[{'requirement_id':'R1','turn':2}] if len(messages)>=3 else [],
            'judgments':decisions,'all_requirements_met':met,'simulated_user_satisfied':satisfied,
            'whole_sample_cap_seconds':a.sample_seconds,'technical_scenario_coverage':False}
        save(private/'episode.json',episode); save(run/'transcript.json',messages)
        (run/'transcript.md').write_text('\n\n'.join('# '+m['message_id']+' ('+m['role']+')\n'+m['text'] for m in messages),encoding='utf-8')
    try:
        fit=llm.json_obj(fit_messages(visible_persona(persona),task),FIT,
            schema_name='roleplay_fit',role='adapter.fit',max_output_tokens=1600)
        validate_fit(fit,task);save(run/'applicability.json',fit)
        if fit['decision']=='explicit_contradiction':
            result={'protocol':PROTOCOL,'task_id':task.task_id,'persona':a.persona,'original_reward':None,
                'online_requirements':[],'objective_requirements':[], 'benchmark_outcome':'invalid',
                'termination':'not_applicable','applicability':fit,'experience_status':'not_executed_optional'}
            return 0
        import docker
        client=docker.from_env(timeout=15)
        try: image=client.images.get(task.docker_image).id
        finally: client.close()
        gold=gold_proof(a.gold_dir,task,image); save(run/'gold-proof.json',gold)
        runtime=Runtime('swe-tb3-'+uuid.uuid4().hex[:12],image,memory='16g',cpus='4',network='none'); runtime.deadline=deadline
        runtime.start(); rc,head,_=runtime.sh('git rev-parse HEAD',timeout=15)
        if rc or head.strip()!=task.base_commit: raise ValueError('Image base commit mismatch')
        snapshots=Snapshots(runtime,task,run)
        agent=TerminalAgent(llm,runtime,json.dumps(task.environment),max_steps_per_turn=14,workdir='/testbed',command_timeout=900)
        message=(a.binding_dir/'instruction.md').read_text(encoding='utf-8')
        senior=a.persona=='senior_race_review_no_tutorial'
        if senior: message+='\nFor this first turn, review only; do not modify tracked or nonignored repository files.'
        # Reserve one of the 24 user messages for a closing acknowledgment.
        for turn in range(1,24):
            if turn==2:
                message+='\n\nThe complete original issue follows; all original acceptance conditions remain binding:\n'+task.instruction
                message+='\n\nYou are now explicitly authorized to implement the local fix in this isolated repository. No commits, remote writes, destructive cleanup, or external services.'
                message+='\n\nAdditional requirement R1 (the full contract):\n'+req['disclosure']
            task.assert_no_leak(message,'user message')
            messages.append({'role':'user','message_id':f'u-{turn}','turn':turn,'text':message}); checkpoint()
            runtime.command_source='agent'; runtime.agent_turn=turn
            try: reply=agent.run_turn(message,turn)
            finally: runtime.command_source='harness'
            messages.append({'role':'agent','message_id':f'a-{turn}','turn':turn,'text':reply.reply,'status':reply.status})
            save(private/f'agent-turn-{turn}.json',reply.to_json()); checkpoint()
            if senior and turn==1:
                _,_,first=snapshots.capture(1)
                if first['patch'].strip(): raise ValueError('Review-only first turn changed the repository')
            if turn<2:
                message='Thank you. Here are the task details and the authorized delivery boundary.'; continue
            current=judge(llm,req,persona,public,messages,turn,audit,judge_prompt,rubric)
            decisions.append({'turn':turn,'stage':'checkpoint','judgment':current}); met=current['verdict']=='MET'; checkpoint()
            if met and reply.status=='done':
                current=judge(llm,req,persona,public,messages,turn,audit,judge_prompt,rubric,final=True)
                decisions.append({'turn':turn,'stage':'final_recheck','judgment':current}); met=current['verdict']=='MET'
            decision=llm.json_obj([{'role':'system','content':simulator_prompt if simulator_prompt.lstrip().startswith('## Response envelope') else SIMULATOR_ENVELOPE+'\n'+simulator_prompt},
                {'role':'user','content':json.dumps({'persona':visible_persona(persona),'public_task':public,'transcript':messages,
                 'current_requirement':req,'judgment':current,'agent_requests_completion':reply.status=='done'},ensure_ascii=False)}],
                SIM_SCHEMA,schema_name='requirement_user',role=f'simulator.turn{turn+1}',max_output_tokens=1000)
            satisfied=bool(met and reply.status=='done' and decision['satisfied'])
            checkpoint()
            if satisfied:
                messages.append({'role':'user','message_id':f'u-close-{turn}','turn':turn,'text':decision['message'],'closing':True})
                messages[-1]['text']='Thank you. I am satisfied with the visible handoff; independent verification is still pending.'
                break
            message=decision['message']
            if not met:
                message+='\nR1 is still '+current['verdict']+': '+current['feedback']
        checkpoint()
        _,_,item=snapshots.capture(99); runtime.remove(); runtime=None
        verifier=Checkers(snapshots,llm,run,task); verifier.protect_official_tests=True; verifier.review_deadline=deadline
        status,reason,evidence=verifier.official(item['patch'])
        original=None if status=='unknown' else int(status=='pass')
        check=checker(req,eligible(messages,2)); save(run/'persona-check.json',{'checks':[check],'independent':True})
        outcome=benchmark_outcome(original,[check],met and satisfied)
        result={'protocol':PROTOCOL,'task_id':task.task_id,'persona':a.persona,'original_reward':original,
            'online_requirements':decisions,'objective_requirements':[check],'simulated_user_satisfied':satisfied,
            'benchmark_outcome':outcome,'official':evidence,'official_reason':reason,
            'experience_status':'not_executed_optional','binding_sha256':file_hash(bound/'binding.json'),
            'termination':'satisfied' if satisfied else 'turn_budget_exhausted','measurement_scope':req['coverage']}
    except Exception as error:
        checkpoint()
        result={'protocol':PROTOCOL,'task_id':task.task_id,'persona':a.persona,'original_reward':None,
            'online_requirements':decisions,'objective_requirements':[], 'benchmark_outcome':'infrastructure_error',
            'error_type':type(error).__name__,'error':str(error)[:1000],'experience_status':'not_executed_optional'}
    finally:
        if runtime is not None:
            try: runtime.remove()
            except Exception as error:
                if result is not None:
                    result.update(benchmark_outcome='infrastructure_error',cleanup_error=type(error).__name__)
        if result is not None:
            save(run/'persona-result.json',result)
            valid=result['benchmark_outcome'] in ('pass','fail')
            inapplicable=result.get('termination')=='not_applicable'
            save(run/'episode.json',{'protocol':PROTOCOL,'metrics':{'base_task_success':result['original_reward'],
                'joint_success':int(result['benchmark_outcome']=='pass') if valid else None},'outcome':result})
            save(run/'release_check.json',{'ready_for_review':valid or inapplicable,
                'result_status':'not_applicable' if inapplicable else 'completed' if valid else 'evaluation_invalid',
                'publication_status':'requires_semantic_review','protocol':PROTOCOL})
    return 0 if result['benchmark_outcome'] in ('pass','fail') else 2


if __name__=='__main__': raise SystemExit(main())
