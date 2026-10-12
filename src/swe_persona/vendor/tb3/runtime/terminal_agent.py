"""Terminal agent with persona reviews and a wall-clock interaction deadline."""
from pathlib import Path
import asyncio, base64, hashlib, json, mimetypes, subprocess, time
from harbor.agents.base import BaseAgent
from azure_client import call, obj, TEXT
from protocol import due_requirement, visible_transcript, MAX_USER_TURNS, write
from persona_adapter import ReactiveUser
from src.personas.salience import activate_persona_preferences
from requirement_loop import RequirementLoop, load_bundle, assess, visible_ledger, POLICY

ACTION=obj({'action':{'type':'string','enum':['shell','reply','ready','view_image','status']},'command':TEXT,'message':TEXT,'image_path':TEXT})
USER=obj({'message':TEXT})

class InteractionDeadline(Exception):
    """The complete interaction, including simulator waits, reached its limit."""

class RequirementJudgeError(Exception):
    """Unavailable/invalid review is infrastructure failure, never task rejection."""

def prompt_messages(messages):
    """Keep every user contract; abbreviate only old tool records if needed."""
    if len(json.dumps(messages,ensure_ascii=False))<=300000:return messages
    result=[]
    for i,m in enumerate(messages):
        if i==0 or i>=len(messages)-40 or (m['role']=='user' and not m['content'].startswith('Tool result: ')):
            result.append(m);continue
        text=m['content']
        if len(text)>1000:text=text[:500]+'\n[Old tool/action text abbreviated; full record archived. SHA256 '+hashlib.sha256(text.encode()).hexdigest()+'. Reinspect task files if needed.]\n'+text[-250:]
        result.append({'role':m['role'],'content':text})
    if len(json.dumps(result,ensure_ascii=False))>700000:raise RuntimeError('Prospective context guard reached; excluded from timeout selection')
    return result

async def docker(*args):
    p=await asyncio.create_subprocess_exec('docker',*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    out,err=await p.communicate()
    if p.returncode:raise RuntimeError(err.decode()[-1500:])
    return out.decode()

class PersonaTerminalAgent(BaseAgent):
    @staticmethod
    def name():return 'persona-terminal-astra'
    def version(self):return 'tb3-v6-requirement-loop-v1'
    def __init__(self,*args,arm='single_turn',limit_seconds=7200,binding_path=None,gpu_device=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.private_dir=self.logs_dir.with_name(self.logs_dir.name+'-private')
        self.arm=arm;self.limit=float(limit_seconds);self.binding=json.loads(Path(binding_path).read_text()) if binding_path else None
        self.events=[];self.ids={};self.expired=False;self.paused=False;self.user_wait=False;self.start=0.;self.end=None;self.freeze=None
        self.user_wait_seconds=0.;self.cutoff_at=None
        self.disclosed=0;self.work_after_disclosure=0
        self.disclosed_ids=[];self.fact_ids=[];self.user_turns=0;self.approved_command=None
        self.plan_approved=False;self.actions_since_user=0;self.visible_activity=[];self.requirement_work={}
        self.user=None;self.persona_evidence=[];self.checkpoints=[];self.simulator_steps=0
        self.requirement_loop = (RequirementLoop(self.binding, load_bundle(self.binding, binding_path))
            if self.binding and arm in {'persona', 'persona_static'} else None)
        self.pause_lock=asyncio.Lock()
        self.paused_at=None
        self.gpu_device=gpu_device
    async def setup(self,environment):
        self.environment=environment
        r=await environment._run_docker_compose_command(['ps','-aq'])
        ids=r.stdout.split()
        details=json.loads(await docker('inspect',*ids))
        self.ids={x['Config']['Labels'].get('com.docker.compose.service','main'):x['Id'] for x in details}
        assert 'main' in self.ids
        main_info=next(x for x in details if x['Id']==self.ids['main'])
        self.runtime_image=main_info['Image']
        # Harbor removes Compose-local tags at teardown. Keep the exact base
        # image available for the independent, post-teardown persona checker.
        await docker('tag', self.runtime_image, 'swe-persona-runtime:'+self.runtime_image.split(':',1)[1])
        self.checker_gpu_requests=main_info['HostConfig'].get('DeviceRequests') or []
        assert next(x for x in details if x['Id']==self.ids['main'])['State']['Running']
        if self.gpu_device is not None:
            main=next(x for x in details if x['Id']==self.ids['main'])
            requests=main['HostConfig'].get('DeviceRequests') or []
            assert len(requests)==1 and requests[0].get('DeviceIDs')==[str(self.gpu_device)],'Unexpected GPU visibility'
    def remaining(self):return max(0.,self.limit-(time.monotonic()-self.start))
    async def within_budget(self,operation):
        try:
            return await asyncio.wait_for(operation,timeout=max(.000001,self.remaining()))
        except asyncio.TimeoutError:
            if self.expired or self.remaining()<=0:
                self.expired=True
                await self.pause(deadline=True)
                raise InteractionDeadline() from None
            raise
    def event(self,kind,data):
        self.events.append({'timestamp':time.time(),'kind':kind,'data':data})
        self.save()
    def save(self):
        self.private_dir.mkdir(parents=True,exist_ok=True)
        elapsed=(self.end or self.cutoff_at or time.monotonic())-self.start if self.start else 0
        value={'arm':self.arm,'time_limit_seconds':self.limit,'clock':'agent_user_wall','events':self.events,
               'excluded_user_seconds':0.,'user_wait_seconds':self.user_wait_seconds,'terminal':getattr(self,'terminal','running'),
               'charged_seconds':elapsed,'interaction_seconds':elapsed,
               'deadline_exceeded':self.expired,
               'freeze':self.freeze,'disclosed_requirements':self.disclosed,'work_actions_after_disclosure':self.work_after_disclosure,
               'disclosed_requirement_ids':self.disclosed_ids,'disclosed_fact_ids':self.fact_ids,'user_turns':self.user_turns,
               'requirement_work_actions':self.requirement_work,'persona_evidence':self.persona_evidence,'checkpoints':self.checkpoints}
        if self.requirement_loop:
            value['requirement_loop'] = self.requirement_loop.snapshot()
        tmp=self.private_dir/'episode.json.tmp';tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(self.private_dir/'episode.json')
    async def pause(self,deadline=False):
        async with self.pause_lock:
            expected=self.start+self.limit
            if not self.paused:
                states=json.loads(await docker('inspect',*self.ids.values()))
                active=[x['Id'] for x in states if x['State']['Running'] and not x['State']['Paused']]
                if active:await docker('pause',*active)
                self.paused=True;self.paused_at=time.monotonic()
            if deadline and self.cutoff_at is None:
                # A simulator wait may already have paused the services earlier.
                # The interaction still ends at the current wall-clock deadline.
                self.cutoff_at=time.monotonic()
                self.freeze={'expected_monotonic':expected,'observed_monotonic':self.cutoff_at,
                             'delay_seconds':max(0.,self.cutoff_at-expected),'containers':list(self.ids.values())}
                self.freeze['valid']=self.freeze['delay_seconds']<=2
    async def resume_containers(self):
        async with self.pause_lock:
            if self.paused and not self.expired and (self.end is not None or self.remaining()>0):
                states=json.loads(await docker('inspect',*self.ids.values()))
                paused=[x['Id'] for x in states if x['State']['Paused']]
                if paused:await docker('unpause',*paused)
                self.paused=False;self.paused_at=None
    async def watch(self):
        while self.end is None:
            if self.remaining()<=0:
                self.expired=True;await self.pause(deadline=True);self.event('budget_exhausted',self.freeze);return
            await asyncio.sleep(.05 if not self.user_wait else .2)
    async def checkpoint(self, label):
        """Keep immutable original-deliverable snapshots; never feed them to the user."""
        await self.pause()
        root=self.private_dir/'checkpoints'/f'{len(self.checkpoints):02d}-{label}';root.mkdir(parents=True,exist_ok=True)
        paths=self.binding.get('original_artifacts',[]) if self.binding else []
        manifest=[]
        for entry in paths:
            raw=entry if isinstance(entry,str) else entry['source']
            service='main' if isinstance(entry,str) else entry.get('service') or 'main'
            if raw.startswith('/logs/') or service not in self.ids:
                manifest.append({'source':raw,'service':service,'status':'unavailable','reason':'Collection requires a log convention or unavailable service'})
                continue
            target=root/raw.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
            try:
                await self.within_budget(docker('cp',self.ids[service]+':'+raw,str(target)))
                patterns=[] if isinstance(entry,str) else entry.get('exclude',[])
                if target.is_dir() and not target.is_symlink() and patterns:
                    import fnmatch,shutil
                    for p in sorted(target.rglob('*'),reverse=True):
                        rel=p.relative_to(target).as_posix()
                        if any(fnmatch.fnmatch(rel,pattern) or fnmatch.fnmatch(p.name,pattern) for pattern in patterns):
                            if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
                            else:p.unlink(missing_ok=True)
                manifest.append({'source':raw,'service':service,'status':'copied'})
            except InteractionDeadline:raise
            except Exception as exc:manifest.append({'source':raw,'service':service,'status':'unavailable','reason':str(exc)[-400:]})
        record={'label':label,'path':str(root),'interaction_seconds':self.limit-self.remaining(),
                'disclosed_requirement_ids':list(self.disclosed_ids),'manifest':manifest,'native_grade':'not_yet_run'}
        self.checkpoints.append(record);write(root/'checkpoint.json',record);self.save()

    async def review_requirements(self, event):
        loop = self.requirement_loop
        if not loop or event not in {'validation_report', 'completion_attempt', 'work_checkpoint'}:
            return
        # Final submissions recheck all already disclosed requirements for regressions.
        rids = ([rid for rid, state in loop.states.items() if state['state'] != 'undisclosed']
                if event == 'completion_attempt' else ([loop.current()] if loop.current() else []))
        for rid in rids:
            state = loop.states[rid]
            if event != 'completion_attempt':
                fresh = [e for e in visible_ledger(self.events) if e['role'] == 'agent'
                         and e['event_index'] > max(state['disclosed_at'], state['reviewed_through'])]
                if not fresh:
                    continue
            self.user_wait=True; start=time.monotonic()
            try:
                result = await self.within_budget(asyncio.to_thread(
                    assess, self.binding, loop, rid, self.events, self.private_dir/'requirement-judge-audit',
                    min(600, self.remaining()), completion=event == 'completion_attempt'))
                loop.record(rid, result, visible_ledger(self.events))
                self.event('requirement_review', result)
            except InteractionDeadline:
                raise
            except Exception as exc:
                raise RequirementJudgeError(str(exc)) from exc
            finally:
                self.user_wait_seconds += max(0., (self.cutoff_at or time.monotonic()) - start)
                self.user_wait=False
            if result['verdict'] != 'MET':
                break

    async def user_response(self,messages,event='clarification',action=None):
        turn_limit = POLICY['max_user_turns'] if self.requirement_loop else MAX_USER_TURNS
        step_limit = POLICY['max_simulator_steps'] if self.requirement_loop else 40
        if self.user_turns>=turn_limit or self.simulator_steps>=step_limit:
            self.terminal='turn_limit';return False
        await self.pause()
        if self.expired or self.remaining()<=0:raise InteractionDeadline()
        turn=self.user_turns+1
        requirements=self.binding['proposal']['requirements']
        await self.review_requirements(event)
        due=(self.requirement_loop.due(turn,event) if self.requirement_loop
             else due_requirement(requirements,self.disclosed_ids,turn,event))
        if due:await self.checkpoint('before-'+due['id'])
        remaining_facts=[f for f in self.binding['public_contract']['facts'] if f['id'] not in self.fact_ids]
        # Original acceptance facts cannot remain withheld when implementation or
        # completion starts. Ordinary clarifications may reveal a focused subset.
        force_facts=event in {'first_plan','work_checkpoint','validation_report','completion_attempt'} or turn>=3
        output=(action or {}).get('message','')
        proposed=(action or {}).get('command') if (action or {}).get('action')=='reply' else None
        active=activate_persona_preferences(self.binding['persona'],agent_output=output,
            agent_activity=' '.join(self.visible_activity),completion_attempt=event=='completion_attempt',
            elapsed_sec=self.limit-self.remaining(),command_gate_mode='proposal_only' if proposed else '')
        end_allowed=event=='completion_attempt' and due is None and not remaining_facts and len(self.disclosed_ids)==len(requirements)
        if self.requirement_loop:
            end_allowed = end_allowed and self.requirement_loop.complete()
        current_id = self.requirement_loop.current() if self.requirement_loop else None
        current_review = self.requirement_loop.states[current_id]['review'] if current_id else None
        corrective = bool(current_review and current_review['verdict'] != 'MET')
        observation={'phase':'followup','rules':self.user.rules(),'turn':turn,'event':event,
            'remaining_seconds':self.remaining(),'active_preferences':active,
            'visible_conversation':visible_transcript(self.events)[-24:],
            'agent_message':output,'activity':list(self.visible_activity),'proposed_command':proposed,
            'allowed_actions':(['message'] if due or (force_facts and remaining_facts) else ['message','continue'])+(['done','abandon'] if end_allowed else []),
            'allowed_fact_ids':[f['id'] for f in remaining_facts],'public_facts':remaining_facts,
            'require_all_facts':force_facts,'due_requirement':due,
            'active_requirements':[r for r in requirements if r['id'] in self.disclosed_ids],
            'end_allowed':end_allowed,'command_approval_required':self.confirm_each}
        if self.requirement_loop:
            observation.update(current_requirement_id=current_id,
                current_requirement_review=current_review, corrective_request_required=corrective,
                requirement_progress={rid: s['state'] for rid, s in self.requirement_loop.states.items()
                                      if s['state'] != 'undisclosed'})
        self.event('user_observation',observation)
        self.user_wait=True;start=time.monotonic()
        try:
            self.user.timeout=min(600,self.remaining())
            decision=await self.within_budget(asyncio.to_thread(self.user.step,observation))
        finally:
            self.user_wait_seconds+=max(0.,(self.cutoff_at or time.monotonic())-start)
            self.user_wait=False
        if self.expired or self.remaining()<=0:raise InteractionDeadline()
        if corrective and (decision['action'] != 'message' or not decision['message'].strip()):
            self.event('simulator_decision_guard', {'reason': 'Unresolved requirement requires a corrective request',
                                                  'original_decision': decision})
            decision = dict(decision, action='message', message_type='redirect',
                            message=current_review['feedback'])
        allowed_facts={f['id']:f for f in remaining_facts}
        if not set(decision['fact_ids'])<=set(allowed_facts):raise ValueError('Simulator invented fact references')
        if decision['action'] not in observation['allowed_actions']:raise ValueError('Illegal simulator action')
        if decision['approve_command'] and not proposed:raise ValueError('Approval lacks a concrete command')
        # A natural reply plus frozen acceptance facts. The simulator cannot alter
        # predicates, create scope, or silently omit a required disclosure.
        selected=list(allowed_facts) if force_facts else list(dict.fromkeys(decision['fact_ids']))
        content=decision['message'].strip()
        if selected:content+='\n\n'+ '\n\n'.join(allowed_facts[key]['text'] for key in selected)
        if due:
            content+='\n\n'+due['disclosure']
            self.disclosed_ids.append(due['id']);self.disclosed=len(self.disclosed_ids)
            self.requirement_work[due['id']]=0
            if self.requirement_loop:
                self.requirement_loop.disclose(due['id'], len(self.events))
        self.fact_ids.extend(key for key in selected if key not in self.fact_ids)
        if decision['approve_command']:
            self.approved_command=proposed
            self.event('command_approval',{'command_sha256':hashlib.sha256(proposed.encode()).hexdigest(),'user_turn':turn})
        if event=='first_plan' and decision.get('approve_plan'):self.plan_approved=True
        if decision['action'] in {'done','abandon'}:
            self.terminal='ready' if decision['action']=='done' else 'user_abandoned'
        self.simulator_steps+=1
        if content or decision['action'] in {'done','abandon'}:self.user_turns=turn
        self.actions_since_user=0;self.visible_activity=[]
        self.persona_evidence.append({'turn':turn,'event':event,'active_preferences':active,'decision':decision,
                                     'requirement_id':due['id'] if due else None})
        if content:
            self.event('simulator',{'message':content,'requirement_id':due['id'] if due else None,
                                    'fact_ids':selected,'decision':decision,'user_turn':turn})
            messages.append({'role':'user','content':content})
        else:self.event('simulator_noop',{'decision':decision,'user_turn':turn})
        if not self.expired:await self.resume_containers()
        return self.terminal=='running'

    async def run(self,instruction,environment,context):
        self.start=time.monotonic();self.terminal='running';self.confirm_each=False;self.plan_first=False
        self.interactive=self.arm in {'persona','persona_static','vague_no_evolution'}
        if self.interactive:
            if self.arm=='vague_no_evolution':
                self.binding=json.loads(json.dumps(self.binding));self.binding['proposal']['requirements']=[]
            self.user=ReactiveUser(self.binding,self.private_dir/'audit',closed_loop=self.requirement_loop is not None)
            terminal=self.binding['persona']['scenario_specific']['terminal']
            self.confirm_each=terminal['command_confirmation']=='confirm-each'
            self.plan_first=terminal['agent_mode_preference']=='plan-first'
            initial=self.binding['opening_message'];self.user_turns=1
            if self.arm=='persona_static':
                self.disclosed_ids=[r['id'] for r in self.binding['proposal']['requirements']]
                self.disclosed=len(self.disclosed_ids);self.fact_ids=[f['id'] for f in self.binding['public_contract']['facts']]
                self.requirement_work={key:0 for key in self.disclosed_ids}
                if self.requirement_loop:
                    for rid in self.disclosed_ids:
                        self.requirement_loop.disclose(rid, 0, static=True)
                initial+='\n\n'+self.binding['public_contract']['instruction']+'\n\n'+'\n\n'.join(r['disclosure'] for r in self.binding['proposal']['requirements'])
        else:initial=instruction
        messages=[{'role':'user','content':initial}];self.event('initial_instruction',messages[0])
        watchdog=asyncio.create_task(self.watch());images=[]
        system='''You are the evaluated terminal coding agent. Work in the provided task container. The current task is the INPUT JSON that follows this instruction. The examples below are shape samples, not that task.

## Response envelope

The entire assistant message is one JSON object. The first character is `{` and the last character is `}`.

- The object has exactly four keys, in this order: `action`, `command`, `message`, and `image_path`. No extra keys. The action key is named `action`, never `tool`.
- Every key is present on every response. An unused string is `""`.
- Do not put words, Markdown fences, `<tool_call>` tags, or `</think>` tags before or after the object. A sentence followed by JSON is rejected, and the action is not executed.
- `command`, `message`, and `image_path` are JSON strings. A newline in a string is `\\n`, a quote is `\\"`, and a backslash is `\\\\`. A raw line break inside a string is invalid JSON.

## Fields

### action

`action` is `shell`, `reply`, `ready`, `view_image`, or `status`.

- `shell`: the host runs `command`.
- `view_image`: the host reads `image_path`.
- `reply`: ask the user a question, or propose an exact command for approval. Put that command in `command` and the question in `message`. This action does not run the command.
- `status`: send a progress or validation report in `message`.
- `ready`: submit completed work. `message` is the handoff the user sees.

### command, message, image_path

- For `shell`, `command` is the script. One batch only. Set `image_path` to `""`.
- For `view_image`, `image_path` is the file to read, and `command` is `""`.
- For `reply`, `status`, and `ready`, the user-facing text is `message`. Set `image_path` to `""`. Set `command` to `""` unless a `reply` is proposing a command to approve.
- The user sees `message` and high-level tool activity, not raw command output. Put requested evidence in `message`.

## Turn behavior

- Use `reply` to clarify underspecified requirements and approval preferences. Do not guess withheld paths or acceptance details.
- Do not read hidden tests or reference solutions, and do not manipulate verifier infrastructure. Public documentation is allowed. Hidden original tests run after the interaction and do not give repair feedback.
- Every agent request, tool call, and user wait shares one wall-clock budget. Deadline exhaustion fails the whole task.
- Preserve every original requirement as the user clarifies or adds scope. Ask relevant questions, act on user feedback, and report progress at meaningful checkpoints.
- When the user requires confirmation, group related operations into one command batch and propose that batch with `reply` before `shell`.
- If the host reports that the previous response was rejected, the next response is still only the JSON object. Do not explain the error outside it.

## Output examples

{"action":"shell","command":"pwd && ls -la","message":"","image_path":""}

{"action":"reply","command":"","message":"Plan:\\n- Add the header \\"X-Auto\\" after you authorize the edit.\\nI will not edit source until you say so.","image_path":""}
'''
        try:
            for index in range(2048):
                if self.expired or self.remaining()<=0:raise InteractionDeadline()
                prompt=system+'\nINPUT JSON:\n'+json.dumps({'remaining_seconds':self.remaining(),'messages':prompt_messages(messages)},ensure_ascii=False)
                try:action=await self.within_budget(asyncio.to_thread(call,'agent',prompt,ACTION,self.private_dir/'audit',min(600,self.remaining()),images))
                except InteractionDeadline:raise
                except Exception as exc:
                    self.terminal='provider_or_protocol_error';self.event('agent_error',{'type':type(exc).__name__,'message':str(exc)});break
                images=[]
                if self.expired or self.remaining()<=0:raise InteractionDeadline()
                self.event('agent_action',action);messages.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
                kind=action['action'];self.actions_since_user+=1
                if kind in {'ready','reply','status'}:
                    if self.interactive:
                        event='completion_attempt' if kind=='ready' else ('first_plan' if self.plan_first and not self.plan_approved else ('validation_report' if kind=='status' else 'clarification'))
                        if kind=='ready':await self.checkpoint('completion-attempt')
                        if not await self.user_response(messages,event,action):break
                    elif kind=='ready':self.terminal='ready';break
                    elif kind=='reply':messages.append({'role':'user','content':'Use the complete original task and your best judgment; no new requirements.'})
                elif kind=='shell':
                    if self.interactive and ((self.plan_first and not self.plan_approved) or (self.confirm_each and action['command']!=self.approved_command)):
                        self.event('command_blocked',{'reason':'plan_or_exact_command_not_approved'})
                        messages.append({'role':'user','content':'Please propose your approach and this exact command batch using reply first; it has not been approved or run.'})
                        continue
                    self.approved_command=None
                    if self.disclosed:self.work_after_disclosure+=1
                    for key in self.disclosed_ids:self.requirement_work[key]+=1
                    try:
                        result=await self.within_budget(environment.exec(command=action['command'],timeout_sec=max(.05,min(300,self.remaining()))))
                        observation={'return_code':result.return_code,'stdout':(result.stdout or '')[-30000:],'stderr':(result.stderr or '')[-10000:]}
                    except InteractionDeadline:raise
                    except Exception as exc:observation={'error':str(exc),'type':type(exc).__name__}
                    self.event('shell_result',observation);messages.append({'role':'user','content':'Tool result: '+json.dumps(observation,ensure_ascii=False)})
                    self.visible_activity.append('shell')
                elif kind=='view_image':
                    path=self.private_dir/f'image-{index}.bin'
                    try:
                        await self.within_budget(environment.download_file(source_path=action['image_path'],target_path=path))
                        if path.stat().st_size>20*1024*1024:raise ValueError('Image exceeds 20 MB')
                        from PIL import Image
                        with Image.open(path) as im:
                            im.thumbnail((2048,2048));png=path.with_suffix('.png');im.convert('RGB').save(png)
                        images=['data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode()]
                        messages.append({'role':'user','content':'Attached image from '+action['image_path']})
                    except InteractionDeadline:raise
                    except Exception as exc:messages.append({'role':'user','content':'Image tool error: '+str(exc)})
                    self.visible_activity.append('view_image')
                if self.interactive and self.actions_since_user>=8:
                    if not await self.user_response(messages,'work_checkpoint',{'message':'','action':'status'}):break
            else:self.terminal='agent_action_limit'
            if self.terminal=='running':self.terminal='controller_error'
        except InteractionDeadline:
            self.expired=True;self.terminal='budget_exhausted';await self.pause(deadline=True)
        except RequirementJudgeError as exc:
            self.terminal='requirement_judge_error'
            self.event('requirement_judge_error',{'type':type(exc).__name__,'message':str(exc)})
        except Exception as exc:
            self.terminal='controller_error';self.event('controller_error',{'type':type(exc).__name__,'message':str(exc)})
        finally:
            if not self.paused:await self.pause(deadline=self.expired)
            if self.remaining()<=0:
                self.expired=True;self.terminal='budget_exhausted';await self.pause(deadline=True)
            self.end=self.cutoff_at or time.monotonic();watchdog.cancel()
            try:await watchdog
            except asyncio.CancelledError:pass
            self.save();context.metadata={'arm':self.arm,'terminal':self.terminal,'clock':'agent_user_wall','deadline_exceeded':self.expired}
