"""Two public API configurations; grading shares the simulator's API settings."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from jsonschema import Draft202012Validator, ValidationError
from openai import APIStatusError, OpenAI

ROLES = ('agent', 'user_simulator', 'judge')
PUBLIC_ROLES = ('agent', 'user_simulator')

def load_config(path, require_keys=False):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict): raise ValueError('API configuration must be a JSON object')
    if 'judge' in config:
        raise ValueError('Remove the judge section: grading uses the user_simulator model, endpoint, options and API key')
    if set(config) - set(PUBLIC_ROLES): raise ValueError('Only agent and user_simulator sections are allowed')
    for role in PUBLIC_ROLES:
        if role not in config: raise ValueError('Missing API role: ' + role)
        c=config[role]
        if not isinstance(c, dict): raise ValueError(role + ' must be a JSON object')
        for key in ['model','base_url','api_key_env','api']:
            if not isinstance(c.get(key),str) or not c[key]: raise ValueError(f'{role}.{key} is required')
        if c['api'] not in {'chat_completions','responses'}: raise ValueError('Unsupported API transport')
        if not c['base_url'].startswith(('https://','http://localhost','http://127.0.0.1')):
            raise ValueError('Use HTTPS or a local API endpoint')
        chat_token_options(c)
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
    return options

COMPLETE_OUTPUT_TOKENS = 16384
_LENGTH_REASONS = {'length', 'max_tokens', 'max_output_tokens'}
OUTPUT_BUDGET_STEPS = (8192, 16384, 32768, 65536, 131072)
_OUTPUT_BUDGET_BY_EFFORT = {
    'none': 8192, 'disabled': 8192, 'minimal': 8192,
    'low': 16384, 'medium': 65536, 'high': 65536,
    'xhigh': 131072, 'max': 131072,
}


def output_budget(effort):
    """Starting completion ceiling for this thinking effort, thinking included."""
    if effort in (None, '', 'inherit'):
        return _OUTPUT_BUDGET_BY_EFFORT['medium']
    return _OUTPUT_BUDGET_BY_EFFORT.get(str(effort).lower(), _OUTPUT_BUDGET_BY_EFFORT['medium'])


def raised_output_budget(current):
    for step in OUTPUT_BUDGET_STEPS:
        if step > current:
            return step
    return None


def lower_output_budget(current):
    previous = OUTPUT_BUDGET_STEPS[0]
    for step in OUTPUT_BUDGET_STEPS:
        if step >= current:
            return previous
        previous = step
    return previous


def generation_timeout(tokens, configured, limit=None):
    configured = float(configured)
    needed = configured if not tokens else min(900.0, max(configured, float(tokens) / 80.0))
    if limit is not None:
        needed = min(needed, float(limit))
    return needed


def token_limit_rejected(message):
    text = (message or '').lower()
    if not any(name in text for name in ('max_tokens', 'max_completion_tokens', 'max_output_tokens')):
        return False
    return any(phrase in text for phrase in ('range', 'maximum', 'must be', 'less than or equal', 'should be'))


def token_range_ceiling(message):
    import re
    text = message or ''
    match = re.search(r'\[\s*\d+\s*,\s*(\d+)\s*\]', text)
    if match:
        return int(match.group(1))
    match = re.search(r'(?:maximum|<=|less than or equal to)\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def weaker_structured_mode(mode):
    """Next output mode when this endpoint rejects or ignores the current one."""
    return {'json_schema': 'json_object', 'json_object': 'prompt'}.get(mode)


def format_rejected(message):
    """True when an HTTP 400 is about the structured-output parameter itself."""
    text = (message or '').lower()
    return any(token in text for token in (
        'response_format', 'json_schema', 'json_object', 'structured output', 'response format'))


def complete_json_object(text):
    """True when the whole assistant message is one JSON object."""
    body = (text or '').strip()
    if not body:
        return False
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


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
        self._structured_floor=None
        self._output_ceiling=None

    def raw(self,messages,schema=None,name='payload',max_tokens=4096,timeout=None):
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
        configured=c.get('structured_output_mode','json_schema' if c.get('strict_json_schema',True) else 'json_object')
        if configured not in {'json_schema','json_object','prompt'}: raise ValueError('Invalid structured_output_mode')
        mode=self._structured_floor or configured
        if role_name(self.role)=='agent' and schema is not None:
            floor=output_budget((c.get('request_options') or {}).get('reasoning_effort'))
            if max_tokens is None or max_tokens<floor: max_tokens=floor
            if self._output_ceiling: max_tokens=min(max_tokens, self._output_ceiling)
        if schema is not None:
            instruction='\nOutput transport: return ONLY one JSON object satisfying this schema, with no prose outside it. Natural-language role content belongs INSIDE the appropriate JSON string fields. Schema: '+json.dumps(schema)
            indices=[i for i,m in enumerate(messages) if m.get('role')=='system' and isinstance(m.get('content'),str)]
            if indices:
                index=indices[-1];messages[index]=dict(messages[index],content=messages[index]['content']+instruction)
            else:messages=[{'role':'system','content':instruction}]+messages
        extra=dict(c.get('request_options',{}))
        if set(extra)&{'model','messages','input','api_key','base_url','text','response_format'}:
            raise ValueError('request_options cannot override routing or output schemas')
        if role_name(self.role)=='agent' and schema is not None:
            for field in ('max_tokens','max_completion_tokens','max_output_tokens'):
                if isinstance(extra.get(field),int) and extra[field]<max_tokens:
                    extra[field]=max_tokens
                if self._output_ceiling and isinstance(extra.get(field),int):
                    extra[field]=min(extra[field], self._output_ceiling)
        started=time.monotonic()
        attempt=mode
        seen=set()
        shrinks=0
        while True:
            if attempt in seen: raise RuntimeError('Structured output mode cycled')
            seen.add(attempt)
            configured_timeout=float(c.get('timeout_sec',120))
            wait=generation_timeout(max_tokens, configured_timeout, timeout) if role_name(self.role)=='agent' and schema is not None else float(timeout or configured_timeout)
            client=self.client.with_options(timeout=max(1.0, wait))
            try:
                if c['api']=='responses':
                    kwargs={'model':c['model'],'input':messages,'max_output_tokens':max_tokens,'store':False,**extra}
                    if schema is not None and attempt!='prompt':
                        kwargs['text']={'format':{'type':'json_schema','name':name,'strict':True,'schema':schema}} if attempt=='json_schema' else {'format':{'type':'json_object'}}
                    response=client.responses.create(**kwargs)
                    text=response.output_text or ''; usage=response.usage
                    input_tokens=getattr(usage,'input_tokens',0);output_tokens=getattr(usage,'output_tokens',0)
                    details=getattr(response,'incomplete_details',None)
                    finish=getattr(details,'reason',None) or ''
                else:
                    token_opts=chat_token_options(c,max_tokens)
                    if role_name(self.role)=='agent' and schema is not None:
                        for field in ('max_tokens','max_completion_tokens'):
                            if isinstance(token_opts.get(field),int) and token_opts[field]<max_tokens:
                                token_opts[field]=max_tokens
                            if self._output_ceiling and isinstance(token_opts.get(field),int):
                                token_opts[field]=min(token_opts[field], self._output_ceiling)
                    kwargs={'model':c['model'],'messages':messages,**token_opts}
                    if schema is not None and attempt!='prompt':
                        kwargs['response_format']={'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':schema}} if attempt=='json_schema' else {'type':'json_object'}
                    response=client.chat.completions.create(**kwargs)
                    text=response.choices[0].message.content or '';usage=response.usage
                    input_tokens=getattr(usage,'prompt_tokens',0);output_tokens=getattr(usage,'output_tokens',0)
                    finish=response.choices[0].finish_reason or ''
            except APIStatusError as exc:
                detail=getattr(exc,'message',None) or str(exc)
                if exc.status_code==400 and token_limit_rejected(detail) and schema is not None and shrinks<4:
                    ceiling=token_range_ceiling(detail) or lower_output_budget(max_tokens)
                    if ceiling and ceiling<max_tokens:
                        self._output_ceiling=ceiling if self._output_ceiling is None else min(self._output_ceiling, ceiling)
                        max_tokens=self._output_ceiling
                        for field in ('max_tokens','max_completion_tokens','max_output_tokens'):
                            if isinstance(extra.get(field),int): extra[field]=min(extra[field], max_tokens)
                        shrinks+=1
                        seen.discard(attempt)
                        continue
                nxt=weaker_structured_mode(attempt) if exc.status_code==400 and format_rejected(detail) else None
                if not nxt or nxt in seen: raise
                attempt=nxt
                continue
            if (schema is not None and attempt=='json_schema' and finish not in _LENGTH_REASONS
                    and text.strip() and not complete_json_object(text)):
                attempt='json_object'
                continue
            break
        if attempt!=configured: self._structured_floor=attempt
        if self.log_dir:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            record={'role':self.role,'model':c['model'],'resolved_model':response.model,'api':c['api'],
                    'elapsed_sec':time.monotonic()-started,'input_tokens':input_tokens,'output_tokens':output_tokens,
                    'messages':messages,'output':text}
            with (self.log_dir/'api-calls.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        return {'text':text,'input_tokens':input_tokens,'output_tokens':output_tokens}

    def json(self,messages,schema,name='payload',max_tokens=4096,timeout=None):
        # Format repair never changes the schema or supplies grading hints. All
        # attempts share the caller's timeout and are included in the API audit.
        deadline=time.monotonic()+float(timeout or self.config.get('timeout_sec',120))
        repair_messages=list(messages)
        validator=Draft202012Validator(schema)
        for attempt in range(3):
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('Structured response deadline exhausted')
            text=self.raw(repair_messages,schema,name,max_tokens,remaining)['text'].strip()
            nxt=raised_output_budget(max_tokens)
            if not text and nxt and (not self._output_ceiling or nxt<=self._output_ceiling): max_tokens=nxt
            if text.startswith('```') and text.endswith('```') and '\n' in text:
                text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
            try:
                value=json.loads(text)
                validator.validate(value)
                return value
            except (json.JSONDecodeError,ValidationError) as error:
                detail=error.msg if isinstance(error,json.JSONDecodeError) else str(list(error.absolute_path))+': '+error.message
                if self.log_dir:
                    with (self.log_dir/'format-rejections.jsonl').open('a') as f:
                        f.write(json.dumps({'role':self.role,'schema':name,'attempt':attempt+1,'error':detail})+'\n')
                if attempt==2: raise
                repair_messages=list(messages)+[{'role':'assistant','content':text},
                    {'role':'user','content':'Your response was rejected as invalid JSON or a schema mismatch. '
                     'Return a corrected JSON object satisfying the same schema, with no additional keys. '
                     'Do not change the task or invent evidence. Validation error: '+detail[:2000]}]
