"""Host API shell agent + frozen persona controller on Harbor 0.23 Docker.

This is the explicit swe-together scaffold, not a reproduction of the
historical OpenCode cohort. No provider credentials are sent into the sandbox.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import shlex
import time
import tomllib
import uuid
from pathlib import Path
from openai import APIConnectionError, RateLimitError, InternalServerError

from .api import RoleAPI, load_config, fingerprint
from .binding import validate, PILOT
from .evaluation import seal_evidence, artifact_snapshot
from .simulator import UserSimulator, initial_persona_message, decision_schema
from swe_persona.api_recovery import assess_interrupted, is_api_interruption
from swe_persona.agent_output import action_envelope

PROFILE='swe-together'
AGENT_SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'action':{'type':'string','enum':['command','message']},
    'content':{'type':'string'},'completion_claim':{'type':'boolean'}},
    'required':['action','content','completion_claim']}
AGENT_PROMPT=action_envelope(AGENT_SCHEMA,[
    {'action':'command','content':'pwd','completion_claim':False},
    {'action':'message','content':'I will follow the current task requirements and permissions.','completion_claim':False}], {
    'action':'Choose command or message. command runs content as one shell script in the repository working directory; cwd changes do not persist between commands. message delivers content to the user without running a command.',
    'content':'Always a JSON string. For command, put the script here. For message, put the complete user-facing plan, question, status or handoff here.',
    'completion_claim':'Always a JSON boolean. Use false for inspection, a plan, a question or progress. Use true only with action=message when claiming the work is complete. This is a claim for controller review, not final benchmark acceptance.'})+'''You are a coding agent working in an isolated repository. Respond only with a JSON object containing action (command or message), content (shell command or user-facing message), and completion_claim (boolean). Use command to inspect, edit and test. Shell commands run in the repository working directory; cwd changes do not persist between commands. Use message to ask the user a question, explain a plan or deliver results. Mark completion_claim true only when claiming completion. Follow the user's scope and authorization boundaries. The user may disclose additional requirements at checkpoints. Only actual command outputs establish that a check ran. Do not inspect evaluator assets or attempt to alter the evaluator. Do not claim to have committed, pushed or opened a PR without doing so and being authorized. Give a user-facing progress message after at most 8 shell commands.'''
COLLECT='''import json,pathlib,subprocess
base='__FROZEN_BASE_COMMIT__'
root=pathlib.Path.cwd()
git=['git','-c','safe.directory='+str(root)]
def run(*args):
 p=subprocess.run(git+list(args),text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr[-500:])
 return p.stdout
names=set(run('diff','--name-only',base).splitlines()+run('ls-files','--others','--exclude-standard').splitlines())
a=[{'ref':'host:workspace_change_status','text':run('status','--short') or '(no changes)'},
   {'ref':'host:final_diff','text':run('diff',base)}]
for name in sorted(names):
 p=root/name
 if p.is_file() and not p.is_symlink() and p.stat().st_size<150000:
  try:a.append({'ref':name,'text':p.read_text()})
  except UnicodeError:pass
print(json.dumps(a))
'''

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

async def agent_command(env, command, workdir, limit):
    """Normalize Harbor 0.23's command timeout without hiding other failures."""
    try:
        r = await env.exec(command, cwd=workdir, timeout_sec=limit)
    except (asyncio.TimeoutError, RuntimeError) as exc:
        if not isinstance(exc, asyncio.TimeoutError) and str(exc) != f'Command timed out after {limit} seconds':
            raise
        return 124, ('Host command timeout. Output unavailable; the container command may still be running. '
                     'Inspect its state before retrying.'), True
    return r.return_code, (r.stdout or '') + '\n' + (r.stderr or ''), False


class InteractionStall:
    """Bound silent message-only loops; final verification still grades the work."""
    policy = 'silent-no-progress-v2'

    def __init__(self, threshold=4):
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError('Interaction stall threshold must be a nonnegative integer')
        self.threshold = threshold
        self.key = None
        self.turns = []

    def observe(self, decision, evidence, command_count, turn, *, pending_disclosures=False):
        key = (evidence['snapshot_sha256'], command_count)
        if pending_disclosures or not self.threshold or decision['action'] != 'no-op' or decision.get('kind') != 'wait':
            self.key = None
            self.turns = []
            return False
        if key != self.key:
            self.turns = []
        self.key = key
        self.turns.append(turn)
        return len(self.turns) >= self.threshold


def runtime_hash():
    h=hashlib.sha256()
    root=Path(__file__).parent
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes())
    for p in sorted((PILOT/'full109/bindings').glob('*.json')):h.update(p.name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()

def response_schema(role,payload):
    """The transport uses the same protocol the controller validates."""
    if role=='user_simulator':return decision_schema()
    if role=='interaction_judge':
        item={'type':'object','additionalProperties':False,'properties':{
            'id':{'type':'string','enum':payload['items']},
            'score':{'type':'integer','enum':[0,1,2]},
            'evidence_refs':{'type':'array','items':{'type':'string'}},
            'reason':{'type':'string'}},'required':['id','score','evidence_refs','reason']}
        return {'type':'object','additionalProperties':False,'properties':{
            'items':{'type':'array','items':item}},'required':['items']}
    if role=='requirement_rubric_judge':
        refs=sorted({item['ref'] for group in ('artifacts','messages','executions') for item in payload['evidence'][group]})
        if 'clauses' in payload and 'evidence_excerpt_of_revision' not in payload:
            grounding={'type':'object','additionalProperties':False,'properties':{
                'kind':{'type':'string','enum':['artifact','execution','agent_message']},
                'ref':{'type':'string'},'quote':{'type':'string'}},'required':['kind','ref','quote']}
            clause={'type':'object','additionalProperties':False,'properties':{
                'id':{'type':'string'},'verdict':{'type':'string','enum':['satisfied','violated','unsupported']},
                'grounding':{'type':'array','items':grounding},'reason':{'type':'string'}},
                'required':['id','verdict','grounding','reason']}
            properties={'packet_id':{'type':'string','const':payload['packet_id']},
                        'clauses':{'type':'array','items':clause}}
        else:
            properties={'packet_id':{'type':'string','const':payload['packet_id']},
                'status':{'type':'string','enum':['met','not_met','unknown']},
                'evidence_refs':{'type':'array','items':{'type':'string','enum':refs} if refs else {'type':'string'}},
                'reason':{'type':'string','minLength':1}}
        return {'type':'object','additionalProperties':False,'properties':properties,'required':list(properties)}
    raise ValueError('No output protocol for role: '+role)


class Calls:
    def __init__(self,config,out,deadline=None):self.config=config;self.out=out;self.deadline=deadline
    async def __call__(self,role,system,payload):
        name='user_simulator' if role=='user_simulator' else 'judge'
        timeout=3*float(self.config[name].get('timeout_sec',120))+3
        if self.deadline is not None:timeout=min(timeout,max(.1,self.deadline-time.monotonic()))
        end=time.monotonic()+timeout
        if self.deadline is not None:end=min(end,self.deadline)
        def invoke():
            api=RoleAPI(self.config,name,self.out/'api')
            schema=response_schema(role,payload)
            messages=[{'role':'system','content':system+' Return one JSON object only. Never cite a collection name such as executions; cite only item refs.'},
                      {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
            try:
                # RoleAPI owns bounded transport/output repair with one deadline.
                return api.json(messages,schema,'persona_decision',16000,max(.1,end-time.monotonic()))
            finally:api.client.close()
        return await asyncio.wait_for(asyncio.to_thread(invoke),timeout=timeout+1)

async def episode(binding,task,config,out,image,seconds=1800,max_steps=200,verifier_seconds=600):
    import docker
    import yaml
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig,NetworkMode,NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    stall=InteractionStall(int(os.environ.get('SWE_PERSONA_TOGETHER_STALL_TURNS','4')))
    validate(binding)
    if binding['original_instruction']!=(task/'instruction.md').read_text() or binding['original_user_context']!=(task/'user_simulation_prompt.md').read_text():
        raise ValueError('Binding does not match task inputs')
    out.mkdir(parents=True,exist_ok=False)
    client=docker.from_env(timeout=900)
    try:
        try:img=client.images.get(image)
        except docker.errors.ImageNotFound:img=await asyncio.to_thread(client.images.pull,image)
        workdir=img.attrs['Config'].get('WorkingDir') or '/workspace/repo'
        image_id=img.id
    finally:client.close()
    native_config=tomllib.loads((task/'task.toml').read_text())['environment']
    cpus=int(native_config.get('cpus',4));memory_mb=int(native_config.get('memory_mb',4096))
    envdir=out/'environment';envdir.mkdir()
    owner='swt-'+uuid.uuid4().hex[:12]
    compose={'services':{'main':{'image':image,'entrypoint':['/bin/bash','-c'],
        'command':['sleep infinity'],'working_dir':workdir,'runtime':'runc','network_mode':'none',
        'environment':{'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':''},
        'labels':{'swe_persona.owner':owner}}}}
    (envdir/'docker-compose.yaml').write_text(yaml.safe_dump(compose))
    paths=TrialPaths(out/'harbor');paths.mkdir()
    policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    env=DockerEnvironment(environment_dir=envdir,environment_name=owner,session_id=owner,
        trial_paths=paths,task_env_config=EnvironmentConfig(docker_image=image,cpus=cpus,memory_mb=memory_mb),
        network_policy=policy,phase_network_policies=[policy])
    result={'profile':PROFILE,'task_id':binding['task_id'],'persona_id':binding['persona']['id'],
        'binding_sha256':binding['binding_sha256'],'runtime_sha256':runtime_hash(),
        'api_config_sha256':fingerprint(config),'docker_image_id':image_id,'original_reward':None,
        'benchmark_outcome':'infrastructure_error','persona_check':{},'cpu_only':True,
        'interaction_stall_policy':{'policy':stall.policy,'threshold':stall.threshold,
                                   'enabled':bool(stall.threshold)}}
    final_artifacts_ready=False
    try:
        await asyncio.wait_for(env.start(force_build=False),timeout=600)
        init=await env.exec('git -c safe.directory='+shlex.quote(workdir)+' rev-parse HEAD && git -c safe.directory='+shlex.quote(workdir)+' update-ref refs/tags/SWT_BASE HEAD',cwd=workdir,timeout_sec=30)
        if init.return_code:raise RuntimeError('Repository baseline could not be recorded: '+str(init.stderr))
        base_commit=init.stdout.strip().splitlines()[0]
        if len(base_commit)!=40 or any(c not in '0123456789abcdef' for c in base_commit):raise RuntimeError('Invalid baseline commit')
        collector=COLLECT.replace('__FROZEN_BASE_COMMIT__',base_commit)
        wheels=task/'tests/wheels'
        if wheels.is_dir():
            await env.upload_dir(wheels,'/tmp/swt-verifier-deps')
            setup=await env.exec('cd /tmp/swt-verifier-deps && sha256sum -c SHA256SUMS && python3 -m pip install --no-index --find-links=. pytest==8.3.4',cwd=workdir,timeout_sec=120)
            write(out/'dependency-setup.json',{'returncode':setup.return_code,'stdout':setup.stdout,'stderr':setup.stderr})
            if setup.return_code:raise RuntimeError('Pinned offline dependencies could not be installed')
            await env.exec('rm -rf /tmp/swt-verifier-deps',cwd=workdir,timeout_sec=10)
        from swe_persona.time_limit import begin_interaction, begin_verification
        deadline=begin_interaction(seconds);seconds=deadline-time.monotonic();calls=Calls(config,out,deadline)
        sim=UserSimulator(binding,calls)
        initial=binding['original_instruction'].rstrip()+'\n\n'+initial_persona_message(binding['persona'])
        sim.history=[{'ref':'user-0','role':'user','turn':0,'text':initial}]
        conversation=[{'role':'system','content':AGENT_PROMPT},{'role':'user','content':initial}]
        executions=[];turn=0;commands_since_message=0;stop='step_limit'
        async def evidence():
            r=await env.exec('python3 -c '+shlex.quote(collector),cwd=workdir,timeout_sec=30)
            if r.return_code:raise RuntimeError('Evidence collection failed: '+str(r.stderr)[-500:])
            return seal_evidence(json.loads(r.stdout),sim.history,executions)
        async def code_call():
            api=RoleAPI(config,'agent',out/'api')
            try:
                from jsonschema import ValidationError
                request=list(conversation)
                for attempt in range(3):
                    try:return await asyncio.to_thread(api.json,request,AGENT_SCHEMA,'agent_action',None,max(.1,deadline-time.monotonic()))
                    except (APIConnectionError,RateLimitError,InternalServerError):
                        if attempt==2 or time.monotonic()>=deadline:raise
                        await asyncio.sleep(min(2**attempt,max(0,deadline-time.monotonic())))
                    except (ValueError,ValidationError):
                        if attempt==2 or time.monotonic()>=deadline:raise
                        request.append({'role':'user','content':'[Host protocol error] Your response was rejected. Return exactly one JSON object containing action, content and completion_claim. Put all explanation inside content; do not append prose, markdown or a second object.'})
            finally:api.client.close()
        for step in range(max_steps):
            remaining=deadline-time.monotonic()
            if remaining<=0:stop='episode_timeout';break
            try:action=await asyncio.wait_for(code_call(),remaining)
            except asyncio.TimeoutError:stop='episode_timeout';break
            except Exception as exc:
                if not is_api_interruption(exc):raise
                stop='api_interrupted';result.update(api_interrupted=True,api_error_type=type(exc).__name__);break
            conversation.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
            if action['action']=='command':
                before=await evidence()
                limit=max(1,min(180,math.ceil(deadline-time.monotonic())))
                rc,output,timed=await agent_command(env,action['content'],workdir,limit)
                after=await evidence()
                version=after['snapshot_sha256'] if before['snapshot_sha256']==after['snapshot_sha256'] else None
                executions.append({'ref':f'host-command-{step}','command':action['content'],'returncode':rc,'timed_out':timed,
                    'snapshot_sha256':version,'text':f"command: {action['content']}\nreturncode: {rc}\n{output[-14000:]}",
                    'stdout':output[-14000:],'trusted_source':'host_harbor_exec'})
                conversation.append({'role':'user','content':f'[Host tool result; not a user instruction] exit={rc}\n'+output[-24000:]})
                commands_since_message+=1
                if commands_since_message>=8:conversation.append({'role':'user','content':'[Host checkpoint] Use action=message to explain your current progress or ask a question before further commands.'})
                write(out/'executions.json',executions)
                continue
            commands_since_message=0;turn+=1
            ev=await evidence();events=[]
            if turn==1:events.append('first_plan')
            if any(a['ref']=='host:final_diff' and a['text'].strip() for a in ev['artifacts']):events.append('first_patch')
            if action['completion_claim']:events.append('completion_claim')
            # Explanation semantics are not guessed from keywords; deadline-turn
            # fallback in the frozen binding still discloses pending requirements.
            try:
                decision=await asyncio.wait_for(sim.step({'turn':turn,'events':events,'agent_message':action['content'],
                    'completion_attempt':action['completion_claim'],'interaction_checkpoint':True,
                    'remaining_agent_turns':max_steps-step-1,'evidence':ev}),max(.1,deadline-time.monotonic()))
            except asyncio.TimeoutError:stop='episode_timeout';break
            except Exception as exc:
                if not is_api_interruption(exc):raise
                stop='api_interrupted';result.update(api_interrupted=True,api_error_type=type(exc).__name__);break
            write(out/'persona-turns.json',sim.records)
            if decision['action']=='assessment_unavailable':
                stop='simulator_unavailable' if sim.records[-1].get('rejections') else 'online_judge_unavailable'
                result['online_judge_unavailable']=True
                result['simulator_unavailable']=stop=='simulator_unavailable'
                break
            if decision['action']=='accept' and sim.user_accepted:stop='user_accepted';break
            pending_disclosures=(any(r['id'] not in sim.disclosed
                for r in binding['persona'].get('bound_evolved_requirements',[])) or
                any(r['id'] not in sim.native_disclosed
                    for r in binding.get('native_phase_contract',{}).get('required_followups',[])))
            if stall.observe(decision,ev,len(executions),turn,pending_disclosures=pending_disclosures):
                stop='interaction_stalled'
                result['interaction_stalled']=True
                write(out/'stall-evidence.json',{
                    'policy':stall.policy,'threshold':stall.threshold,
                    'turns':stall.turns,'snapshot_sha256':ev['snapshot_sha256'],
                    'command_count':len(executions),
                    'reason':'Consecutive simulator wait decisions with no commands or workspace changes.',
                    'grading':'Preserve work and run independent final native/persona verification.'})
                break
            conversation.append({'role':'user','content':decision['content'] if decision['action']=='message' else '[The user is temporarily silent. Continue the authorized task.]'})
        coding_elapsed=seconds-max(0,deadline-time.monotonic())
        # Finalization and native verification are explicitly outside coding time.
        begin_verification()
        calls.deadline=time.monotonic()+180
        ev=await evidence();write(out/'persona-evidence.json',ev)
        write(out/'transcript.json',sim.history)
        write(out/'executions.json',executions)
        final_artifacts_ready=True
        result.update(stop_reason=stop,elapsed_coding_sec=coding_elapsed)
        try:summary=await asyncio.wait_for(sim.finish(ev),180)
        except Exception as exc:
            if not is_api_interruption(exc):raise
            result.update(api_interrupted=True,api_error_type=type(exc).__name__)
            summary={'persona_status':'unknown','evaluation_valid':False,'requirements':[],
                     'user_accepted':sim.user_accepted,'error_type':type(exc).__name__}
        write(out/'persona-summary.json',summary);write(out/'transcript.json',sim.history)
        met=summary['persona_status']=='met';valid=summary['evaluation_valid']
        result.update(persona_check={'status':'pass' if met else 'fail' if summary['persona_status']=='not_met' else 'unknown'},
                      persona_status=summary['persona_status'],user_accepted=summary['user_accepted'])
        # Stop the agent and its background processes. Grade a snapshot in a
        # fresh container with read-only protected tests, never in the live sandbox.
        def verify_snapshot():
            client=docker.from_env(timeout=verifier_seconds+60)
            snapshot=None;verifier=None
            try:
                running=client.containers.list(filters={'label':'swe_persona.owner='+owner})
                if len(running)!=1:raise RuntimeError('Cannot identify owned agent container')
                snapshot=running[0].commit(pause=True)
                running[0].stop(timeout=5)
                logs=out/'native-verifier';logs.mkdir();logs.chmod(0o777)
                verifier=client.containers.run(snapshot.id,entrypoint=['/bin/bash','-c'],
                    command=['mkdir -p /logs/verifier; bash /tests/test.sh'],detach=True,
                    network_disabled=True,runtime='runc',cpu_period=100000,cpu_quota=cpus*100000,
                    mem_limit=str(memory_mb)+'m',working_dir=workdir,
                    environment={'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':''},
                    volumes={str(task/'tests'):{'bind':'/tests','mode':'ro'},str(logs):{'bind':'/logs/verifier','mode':'rw'}},
                    labels={'swe_persona.verifier':owner})
                waited=verifier.wait(timeout=verifier_seconds)
                text=verifier.logs().decode(errors='replace')
                (logs/'execution.log').write_text(text)
                reward=float((logs/'reward.txt').read_text().strip())
                if waited['StatusCode']!=0 or not 0<=reward<=1:raise RuntimeError('Native verifier did not complete with a valid reward')
                write(out/'native-execution.json',{'returncode':waited['StatusCode'],'fresh_snapshot':True,'protected_tests_read_only':True})
                return reward
            finally:
                if verifier is not None:verifier.remove(force=True,v=True)
                if snapshot is not None:client.images.remove(snapshot.id,force=True)
                client.close()
        reward=await asyncio.to_thread(verify_snapshot)
        result['original_reward']=reward
        status='pass' if reward==1 and met and valid else 'invalid' if not valid else 'fail'
        if result.get('interaction_stalled') or result.get('api_interrupted') or result.get('online_judge_unavailable') or any(r.get('api_interrupted') for r in summary.get('requirements',[])):
            progress = any(item.get('role') == 'agent' and item.get('text','').strip() for item in sim.history)
            progress = progress or any(item.get('ref') != 'host:execution_index' and
                item.get('text','').strip() not in ('','(no changes)') for item in ev['artifacts'])
            final_requirements={r['requirement_id']:r for r in summary.get('requirements',[])}
            states=[final_requirements.get(r['id'],{}).get('status')
                    for r in binding['persona']['bound_evolved_requirements']]
            missing=summary.get('undisclosed',[])
            persona = (False if 'not_met' in states or missing else
                       True if met and valid and states and all(s=='met' for s in states) else None)
            assessment = assess_interrupted(reward==1,persona,artifacts_available=True,progress=progress,
                                            budget_exhausted=coding_elapsed>=seconds,
                                            integrity=not summary.get('invalid_reasons')
                                            and not summary.get('base_verifier_modified'))
            if result.get('online_judge_unavailable'):
                assessment.update(api_interrupted=False,online_judge_unavailable=True,
                    outcome_reason='online_judge_unavailable',policy='registered-refs-internal-recovery-v2')
            if result.get('interaction_stalled'):
                assessment.update(api_interrupted=bool(result.get('api_interrupted')) or
                    any(r.get('api_interrupted') for r in summary.get('requirements',[])),
                    interaction_stalled=True,outcome_reason=stop,policy=stall.policy)
                result['interaction_stall_assessment']=assessment
            status=assessment['status']
            if assessment.get('api_interrupted') or result.get('online_judge_unavailable'):
                result['api_recovery_assessment']=assessment
        result.update(original_reward=reward,benchmark_outcome=status,stop_reason=stop,
            persona_check={'status':'pass' if met else 'fail' if summary['persona_status']=='not_met' else 'unknown'},
            persona_status=summary['persona_status'],user_accepted=summary['user_accepted'],
            elapsed_coding_sec=coding_elapsed,native_verifier_exit_code=0)
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[:1500])
        if result.get('api_interrupted') or is_api_interruption(exc):
            progress=any(item.get('role')=='agent' and item.get('text','').strip() for item in getattr(locals().get('sim'),'history',[]))
            saved=locals().get('ev',{})
            progress=progress or any(item.get('ref')!='host:execution_index' and item.get('text','').strip() not in ('','(no changes)')
                for item in saved.get('artifacts',[]))
            native=result['original_reward']==1 if result['original_reward'] is not None else None
            assessment=assess_interrupted(native,None,artifacts_available=final_artifacts_ready,progress=progress,
                budget_exhausted=locals().get('coding_elapsed',0)>=seconds)
            result.update(benchmark_outcome=assessment['status'],api_recovery_assessment=assessment)
    finally:
        try:await asyncio.wait_for(env.stop(delete=True),60)
        except Exception as exc:result['cleanup_error']=type(exc).__name__
        write(out/'persona-result.json',result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--image',required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--max-steps',type=int,default=200)
    a=p.parse_args(argv);config=load_config(a.config,require_keys=True);binding=json.loads(a.binding.read_text())
    gate=json.loads(a.calibration.read_text())
    from .calibration_policy import calibration_allowed, mode
    if (not calibration_allowed(gate) or gate.get('mode')!=mode()
            or gate.get('runtime_sha256')!=runtime_hash() or gate.get('api_config_sha256')!=fingerprint(config)):
        raise ValueError('An approved calibration for the exact API configuration and runtime is required')
    if binding['binding_sha256'] not in gate.get('binding_sha256s',[]):raise ValueError('Binding was not covered by calibration')
    result=asyncio.run(episode(binding,PILOT/'SWE-Together/tasks'/binding['task_id'],config,a.output,a.image,a.seconds,a.max_steps))
    print(json.dumps(result,indent=2));return 0 if result['benchmark_outcome'] in {'pass','fail'} else 2

if __name__=='__main__':raise SystemExit(main())
