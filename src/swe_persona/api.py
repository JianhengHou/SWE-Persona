"""Two public API configurations; grading shares the simulator's API settings."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from jsonschema import Draft202012Validator, ValidationError
from openai import APIStatusError, OpenAI
from .api_recovery import append_audit, retry_transport
from .api_output import (POLICY as OUTPUT_POLICY, IncompleteOutputError, OutputBudget,
                         coding_budget, configured_effort, configured_limit, is_truncated)
from .output_mode import POLICY as FORMAT_POLICY, OutputModeFallback, configured_mode, mode_messages

ROLES = ('agent', 'user_simulator', 'judge')
PUBLIC_ROLES = ('agent', 'user_simulator')
DEFAULT_USER_SIMULATOR_CONFIG = {
    'api': 'responses',
    'base_url': 'https://matraix.services.ai.azure.com/openai/v1',
    'model': 'gpt-6-luna',
    'api_key_env': 'SWE_PERSONA_SIMULATOR_API_KEY',
    'strict_json_schema': True,
    'structured_output_mode': 'json_schema',
    'structured_output_fallback': True,
    'timeout_sec': 300,
    'request_options': {'reasoning_effort': 'medium', 'max_output_tokens': 32768},
}

def load_config(path, require_keys=False):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict): raise ValueError('API configuration must be a JSON object')
    if 'judge' in config:
        raise ValueError('Remove the judge section: grading uses the user_simulator model, endpoint, options and API key')
    if set(config) - set(PUBLIC_ROLES): raise ValueError('Only agent and user_simulator sections are allowed')
    if 'user_simulator' not in config:
        config['user_simulator'] = deepcopy(DEFAULT_USER_SIMULATOR_CONFIG)
    for role in PUBLIC_ROLES:
        if role not in config: raise ValueError('Missing API role: ' + role)
        c=config[role]
        if not isinstance(c, dict): raise ValueError(role + ' must be a JSON object')
        for key in ['model','base_url','api_key_env','api']:
            if not isinstance(c.get(key),str) or not c[key]: raise ValueError(f'{role}.{key} is required')
        if c['api'] not in {'chat_completions','responses'}: raise ValueError('Unsupported API transport')
        configured_mode(c)
        if not c['base_url'].startswith(('https://','http://localhost','http://127.0.0.1')):
            raise ValueError('Use HTTPS or a local API endpoint')
        if c['api']=='responses':responses_options(c)
        else:chat_token_options(c)
        if 'api_key' in c: raise ValueError('Use api_key_env; do not save API keys in configuration')
        if require_keys and not os.environ.get(c['api_key_env']): raise ValueError('Unset key variable: '+c['api_key_env'])
    # Source profiles retain separate judge prompts and audit records. Copy the
    # connection settings so per-role state cannot mutate simulator settings.
    config['judge'] = deepcopy(config['user_simulator'])
    return config

def chat_token_options(config, max_tokens=None):
    """Select one provider-supported Chat token field, without silent duplicates."""
    options=dict(config.get('request_options',{}))
    legacy=options.pop('max_tokens_field',None)
    field=config.get('max_tokens_field',legacy or 'max_completion_tokens')
    if field not in {'max_tokens','max_completion_tokens'}:
        raise ValueError('max_tokens_field must be max_tokens or max_completion_tokens')
    if legacy is not None and legacy!=field:
        raise ValueError('Conflicting max_tokens_field settings')
    limits=[options.pop(k) for k in ('max_tokens','max_completion_tokens') if k in options]
    if len(limits)>1:
        raise ValueError('Specify only one Chat token limit in request_options')
    limit=limits[0] if limits else max_tokens
    if limit is not None:
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<=0:
            raise ValueError('Chat token limit must be a positive integer')
        options[field]=limit
    # Z.ai rejects thinking as a top-level Chat field; the OpenAI SDK forwards extra_body.
    thinking=options.pop('thinking',None)
    if thinking is not None:
        body=dict(options.get('extra_body') or {})
        if not isinstance(body,dict):raise ValueError('extra_body must be an object')
        body['thinking']=thinking
        options['extra_body']=body
    return options

def responses_options(config, max_tokens=None):
    """Translate shared reasoning/token options to the Responses API fields."""
    options=deepcopy(config.get('request_options',{}))
    effort=options.pop('reasoning_effort',None)
    if effort is not None:
        reasoning=options.get('reasoning',{})
        if not isinstance(reasoning,dict):raise ValueError('reasoning must be an object')
        if reasoning.get('effort',effort)!=effort:raise ValueError('Conflicting reasoning effort settings')
        options['reasoning']=dict(reasoning,effort=effort)
    options.pop('max_tokens_field',None)
    limits=[options.pop(k) for k in ('max_tokens','max_completion_tokens','max_output_tokens') if k in options]
    if len(limits)>1:raise ValueError('Specify only one Responses token limit in request_options')
    limit=limits[0] if limits else max_tokens
    if limit is not None:
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<=0:raise ValueError('Responses token limit must be a positive integer')
        options['max_output_tokens']=limit
    return options

def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def role_name(role):
    if role in {'agent','coding_agent'} or role.startswith('agent.'): return 'agent'
    if role in {'simulator','user_simulator'} or role.startswith('simulator'): return 'user_simulator'
    return 'judge'

class RoleAPI:
    def __init__(self, config, role, log_dir=None):
        self.role=role;self.config=config[role];self.log_dir=Path(log_dir) if log_dir else None
        c=self.config
        key=os.environ.get(c['api_key_env'])
        if not key: raise ValueError('Unset key variable: '+c['api_key_env'])
        self.client=OpenAI(base_url=c['base_url'],api_key=key,max_retries=0,
                           timeout=float(c.get('timeout_sec',120)))
        self._output_ceiling=None
        self._format=OutputModeFallback(c['base_url'],c['model'],c['api'],configured_mode(c),
                                       c.get('structured_output_fallback',True))

    def output_budget(self, schema, max_tokens=None):
        options=self.config.get('request_options',{})
        if role_name(self.role)=='agent':
            return coding_budget(options,requested=max_tokens,ceiling=self._output_ceiling)
        effort=configured_effort(options)
        return OutputBudget.resolve(adaptive=role_name(self.role)=='agent' and schema is not None,
            effort=effort, configured=configured_limit(options), requested=max_tokens,
            ceiling=self._output_ceiling)

    def raw(self,messages,schema=None,name='payload',max_tokens=None,timeout=None,*,_budget=None,_deadline=None):
        c=self.config; messages=list(messages)
        converted=[]
        for message in messages:
            message=dict(message)
            if isinstance(message.get('content'),list):
                blocks=[]
                for block in message['content']:
                    block=dict(block)
                    if c['api']=='chat_completions' and block.get('type')=='input_text':block['type']='text'
                    elif c['api']=='chat_completions' and block.get('type')=='input_image':
                        block={'type':'image_url','image_url':{'url':block['image_url']}}
                    elif c['api']=='responses' and block.get('type')=='image_url':
                        block={'type':'input_image','image_url':block['image_url']['url']}
                    elif c['api']=='responses' and block.get('type')=='text':block['type']='input_text'
                    blocks.append(block)
                message['content']=blocks
            converted.append(message)
        messages=converted
        mode=self._format.mode
        request_timeout=float(c.get('timeout_sec',120))
        deadline=time.monotonic()+float(timeout if timeout is not None else 3*request_timeout+3)
        if _deadline is not None:deadline=min(deadline,_deadline)
        budget=_budget if _budget is not None else self.output_budget(schema,max_tokens)
        def audit(record):
            if self.log_dir:
                append_audit(self.log_dir/'transport-recovery.jsonl',dict(record,role=self.role,model=c['model']))
        extra=dict(c.get('request_options',{}))
        if set(extra)&{'model','messages','input','api_key','base_url','text','response_format'}:
            raise ValueError('request_options cannot override routing or output schemas')
        started=time.monotonic()
        total_input=total_output=0
        while True:
            request_messages=mode_messages(messages,schema,mode)
            strict=mode=='json_schema'
            options=responses_options(c) if c['api']=='responses' else chat_token_options(c)
            for field in ('max_tokens','max_completion_tokens','max_output_tokens'):options.pop(field,None)
            if c['api']=='responses':
                kwargs={'model':c['model'],'input':request_messages,'store':False,**options,'max_output_tokens':budget.current}
                if schema is not None and strict:kwargs['text']={'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}}
                elif schema is not None and mode=='json_object':kwargs['text']={'format':{'type':'json_object'}}
                operation=lambda seconds:self.client.with_options(timeout=seconds).responses.create(**kwargs)
            else:
                field=c.get('max_tokens_field',extra.get('max_tokens_field','max_completion_tokens'))
                kwargs={'model':c['model'],'messages':request_messages,**options,field:budget.current}
                if schema is not None and mode!='prompt':
                    kwargs['response_format']={'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':schema}} if strict else {'type':'json_object'}
                operation=lambda seconds:self.client.with_options(timeout=seconds).chat.completions.create(**kwargs)
            try:
                response=retry_transport(operation,request_timeout=request_timeout,deadline=deadline,audit=audit)
            except APIStatusError as error:
                old_limit=budget.current
                if budget.shrink(error.status_code,error.message):
                    self._output_ceiling=budget.ceiling
                    audit({'policy':OUTPUT_POLICY,'status':'token_ceiling','from':old_limit,'to':budget.current})
                    continue
                next_mode=self._format.reject(mode,error.status_code,error.message) if schema is not None else None
                if next_mode is None:raise
                audit({'policy':FORMAT_POLICY,'status':'mode_fallback','from':mode,'to':next_mode,
                       'http_status':error.status_code,'reason':'provider_rejected_output_mode',
                       'action_executed':False})
                mode=next_mode
                continue
            usage=response.usage
            if c['api']=='responses':
                text=response.output_text or ''
                input_tokens=getattr(usage,'input_tokens',0);output_tokens=getattr(usage,'output_tokens',0)
                reason=getattr(getattr(response,'incomplete_details',None),'reason',None)
                finish=getattr(response,'status',None)
            else:
                text=response.choices[0].message.content or ''
                input_tokens=getattr(usage,'prompt_tokens',0);output_tokens=getattr(usage,'completion_tokens',0)
                finish=response.choices[0].finish_reason;reason=finish
            total_input+=input_tokens;total_output+=output_tokens
            if self.log_dir:
                token_details=getattr(usage,'completion_tokens_details',None) or getattr(usage,'output_tokens_details',None)
                reasoning_content=getattr(response.choices[0].message,'reasoning_content',None) if c['api']=='chat_completions' else None
                append_audit(self.log_dir/'api-calls.jsonl',{
                    'role':self.role,'model':c['model'],'resolved_model':response.model,'api':c['api'],
                    'elapsed_sec':time.monotonic()-started,'input_tokens':input_tokens,'output_tokens':output_tokens,
                    'requested_max_tokens':budget.current,'finish_reason':finish,'incomplete_reason':reason,
                    'effort':configured_effort(c.get('request_options',{})),
                    'reasoning_tokens':getattr(token_details,'reasoning_tokens',None),
                    'reasoning_content_chars':len(reasoning_content) if isinstance(reasoning_content,str) else None,
                    'visible_output_chars':len(text),
                    'structured_output_mode':mode,'configured_output_mode':self._format.initial,
                    'messages':request_messages,'output':text})
            old_limit=budget.current
            if budget.grow(reason):
                audit({'policy':OUTPUT_POLICY,'status':'length_retry','reason':reason,
                       'from':old_limit,'to':budget.current,'action_executed':False})
                continue
            if (schema is not None or role_name(self.role)=='agent') and is_truncated(reason) and _budget is None:
                # Raw text-agent bridges parse the returned text without carrying
                # finish metadata. Never let them execute a truncated action.
                raise IncompleteOutputError('Incomplete agent response: '+reason)
            return {'text':text,'input_tokens':total_input,'output_tokens':total_output,
                    'incomplete_reason':reason,'requested_max_tokens':budget.current}

    def json(self,messages,schema,name='payload',max_tokens=None,timeout=None):
        # Format repair never changes the schema or supplies grading hints. All
        # attempts share the caller's timeout and are included in the API audit.
        deadline=time.monotonic()+float(timeout if timeout is not None else 3*float(self.config.get('timeout_sec',120))+3)
        repair_messages=list(messages)
        validator=Draft202012Validator(schema)
        budget=self.output_budget(schema,max_tokens)
        for attempt in range(3):
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('Structured response deadline exhausted')
            result=self.raw(repair_messages,schema,name,max_tokens,remaining,_budget=budget,_deadline=deadline)
            text=result['text'].strip()
            if text.startswith('```') and text.endswith('```') and '\n' in text:
                text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
            try:
                if is_truncated(result.get('incomplete_reason')):
                    raise json.JSONDecodeError('Incomplete response: '+result['incomplete_reason'],text,0)
                value=json.loads(text)
                validator.validate(value)
                return value
            except (json.JSONDecodeError,ValidationError) as error:
                detail=error.msg if isinstance(error,json.JSONDecodeError) else str(list(error.absolute_path))+': '+error.message
                if self.log_dir:
                    with (self.log_dir/'format-rejections.jsonl').open('a') as f:
                        f.write(json.dumps({'role':self.role,'schema':name,'attempt':attempt+1,'error':detail})+'\n')
                if attempt==2: raise
                repair_messages=list(messages)
                if text:
                    repair_messages.append({'role':'assistant','content':text})
                repair_messages.append(
                    {'role':'user','content':'Your response was rejected as invalid JSON or a schema mismatch. '
                     'Return a corrected JSON object satisfying the same schema, with no additional keys. '
                     'Do not change the task or invent evidence. Validation error: '+detail[:2000]})
