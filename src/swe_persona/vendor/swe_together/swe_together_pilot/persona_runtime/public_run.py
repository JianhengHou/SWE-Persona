"""Host API shell agent + frozen persona controller on Harbor 0.23 Docker.

This is the explicit swe-together scaffold, not a reproduction of the
historical OpenCode cohort. No provider credentials are sent into the sandbox.
"""
import argparse
import asyncio
import hashlib
import json
import math
import shlex
import time
import tomllib
import uuid
from pathlib import Path
from openai import APIConnectionError, RateLimitError, InternalServerError

from .api import RoleAPI, load_config, fingerprint
from .binding import validate, PILOT
from .evaluation import seal_evidence, artifact_snapshot
from .simulator import UserSimulator, initial_persona_message

PROFILE='swe-together'
AGENT_SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'action':{'type':'string','enum':['command','message']},
    'content':{'type':'string'},'completion_claim':{'type':'boolean'}},
    'required':['action','content','completion_claim']}
AGENT_PROMPT='''You are a coding agent working in an isolated repository. Task instructions and permissions come from the user message.

## Response envelope

The entire assistant message is one JSON object. The first character is `{` and the last character is `}`.

- The object has exactly three keys, in this order: `action`, `content`, and `completion_claim`. No extra keys.
- Do not put words, Markdown fences, `<tool_call>` tags, or `</think>` tags before or after the object. A sentence followed by JSON is rejected, and the action is not executed.
- `content` is a JSON string. A newline in that string is `\\n`, a quote is `\\"`, and a backslash is `\\\\`. A raw line break inside the string is invalid JSON.
- `completion_claim` is a JSON boolean, `false` or `true`, never a string.

## Fields

### action

`action` is `command` or `message`.

- `command`: the host runs `content` as one shell script in the repository working directory. Directory changes do not persist between commands.
- `message`: the host does not run a command. `content` is delivered to the user.

### content

`content` is always a string.

- For `command`, `content` is the script itself. One script only. Do not describe the script outside the object.
- For `message`, `content` is the whole message the user sees. Put the plan, questions, and status here, with line breaks escaped inside this string. The user never sees text outside the JSON object.

### completion_claim

Set `completion_claim` to `true` only when this message claims the work is complete. Inspection, a plan, a question, and a progress note use `false`.

## Turn behavior

- Follow the user's scope and authorization boundaries. The user may disclose additional requirements at checkpoints.
- Only an actual command output establishes that a check ran.
- Do not inspect evaluator assets or attempt to alter the evaluator.
- Do not claim to have committed, pushed, or opened a PR without doing so and being authorized.
- After at most 8 shell commands, send a user-facing progress message before further commands.
- If the host reports that the previous response was rejected, the next response is still only the JSON object. Do not explain the error outside it.

## Output examples

{"action":"command","content":"ls -la","completion_claim":false}

{"action":"message","content":"Plan:\\n- Read routing.py.\\n- After you authorize edits, add the header \\"X-Auto\\".\\nI will not edit source until you say so.","completion_claim":false}
'''
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

def runtime_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file() and p.suffix in {'.py','.json'}:h.update(p.name.encode()+b'\0'+p.read_bytes())
    for p in sorted((PILOT/'full109/bindings').glob('*.json')):h.update(p.name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()

class Calls:
    def __init__(self,config,out,deadline=None):self.config=config;self.out=out;self.deadline=deadline
    async def __call__(self,role,system,payload):
        name='user_simulator' if role=='user_simulator' else 'judge'
        timeout=float(self.config[name].get('timeout_sec',120))
        if self.deadline is not None:timeout=min(timeout,max(.1,self.deadline-time.monotonic()))
        def invoke():
            api=RoleAPI(self.config,name,self.out/'api')
            schema={'type':'object'}
            if role=='requirement_rubric_judge':
                refs=sorted({item['ref'] for group in ('artifacts','messages','executions') for item in payload['evidence'][group]})
                schema={'type':'object','additionalProperties':False,'properties':{
                    'packet_id':{'type':'string','const':payload['packet_id']},
                    'status':{'type':'string','enum':['met','not_met','unknown']},
                    'evidence_refs':{'type':'array','items':{'type':'string','enum':refs} if refs else {'type':'string'}},
                    'reason':{'type':'string','minLength':1}},
                    'required':['packet_id','status','evidence_refs','reason']}
            messages=[{'role':'system','content':system+' Return one JSON object only. Never cite a collection name such as executions; cite only item refs.'},
                      {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
            end=time.monotonic()+timeout
            try:
                from jsonschema import ValidationError
                for attempt in range(3):
                    try:return api.json(messages,schema,'persona_decision',16000,max(.1,end-time.monotonic()))
                    except (APIConnectionError,RateLimitError,InternalServerError):
                        if attempt==2 or time.monotonic()>=end:raise
                        time.sleep(min(2**attempt,max(0,end-time.monotonic())))
                    except (ValueError,ValidationError) as exc:
                        if attempt==2 or time.monotonic()>=end:raise
                        messages.append({'role':'user','content':'Your previous response was rejected as invalid JSON or invalid schema. Return exactly one JSON object; copy packet_id exactly and use only item refs listed in the schema. Do not add commentary or executable expressions.'})
            finally:api.client.close()
        return await asyncio.wait_for(asyncio.to_thread(invoke),timeout=timeout+1)

async def episode(binding,task,config,out,image,seconds=1800,max_steps=200,verifier_seconds=600):
    import docker
    import yaml
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig,NetworkMode,NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
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
        'benchmark_outcome':'infrastructure_error','persona_check':{},'cpu_only':True}
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
        deadline=time.monotonic()+seconds;calls=Calls(config,out,deadline)
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
                    try:return await asyncio.to_thread(api.json,request,AGENT_SCHEMA,'agent_action',16000,max(.1,min(float(config['agent'].get('timeout_sec',120)),deadline-time.monotonic())))
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
            conversation.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
            if action['action']=='command':
                before=await evidence()
                limit=max(1,min(180,math.ceil(deadline-time.monotonic())))
                timed=False
                try:r=await env.exec(action['content'],cwd=workdir,timeout_sec=limit);rc=r.return_code;output=(r.stdout or '')+'\n'+(r.stderr or '')
                except asyncio.TimeoutError:timed=True;rc=124;output='Host command timeout.'
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
            write(out/'persona-turns.json',sim.records)
            if decision['action']=='accept' and sim.user_accepted:stop='user_accepted';break
            conversation.append({'role':'user','content':decision['content'] if decision['action']=='message' else '[The user is temporarily silent. Continue the authorized task.]'})
        coding_elapsed=seconds-max(0,deadline-time.monotonic())
        # Finalization and native verification are explicitly outside coding time.
        calls.deadline=time.monotonic()+180
        ev=await evidence();write(out/'persona-evidence.json',ev)
        summary=await asyncio.wait_for(sim.finish(ev),180)
        write(out/'persona-summary.json',summary);write(out/'transcript.json',sim.history)
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
        met=summary['persona_status']=='met';valid=summary['evaluation_valid']
        status='pass' if reward==1 and met and valid else 'invalid' if not valid else 'fail'
        result.update(original_reward=reward,benchmark_outcome=status,stop_reason=stop,
            persona_check={'status':'pass' if met else 'fail' if summary['persona_status']=='not_met' else 'unknown'},
            persona_status=summary['persona_status'],user_accepted=summary['user_accepted'],
            elapsed_coding_sec=coding_elapsed,native_verifier_exit_code=0)
    except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc)[:1500])
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
    if not gate.get('passed') or gate.get('runtime_sha256')!=runtime_hash() or gate.get('api_config_sha256')!=fingerprint(config):raise ValueError('A passed calibration for the exact API configuration and runtime is required')
    if binding['binding_sha256'] not in gate.get('binding_sha256s',[]):raise ValueError('Binding was not covered by calibration')
    result=asyncio.run(episode(binding,PILOT/'SWE-Together/tasks'/binding['task_id'],config,a.output,a.image,a.seconds,a.max_steps))
    print(json.dumps(result,indent=2));return 0 if result['benchmark_outcome'] in {'pass','fail'} else 2

if __name__=='__main__':raise SystemExit(main())
