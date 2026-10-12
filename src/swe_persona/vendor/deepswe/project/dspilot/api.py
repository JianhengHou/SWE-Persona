"""Shared host-only model API for agent, simulator and judge.

OpenAI and Azure v1 use Responses or Chat Completions; callers always receive
Responses-shaped results. Credentials never enter task containers.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from .api_config import config, role_config, public_routes
from .api_codec import (_LENGTH_REASONS, chat_payload, chat_response, complete_json_object,
                        format_rejected, generation_timeout, lower_output_budget, output_budget,
                        token_limit_rejected, token_range_ceiling, weaker_structured_mode)
from .coding_budget import bounded_timeout

ROOT = Path(__file__).resolve().parents[1]


class ApiFailure(RuntimeError):
    pass


class Client:
    def __init__(self, log: Path | None = None, max_calls: int = 100, deadline: float | None = None,
                 *, env_file: Path | None = None):
        self.cfg = config(env_file) if env_file is not None else config()
        self.log = log
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_tokens': 0}
        self._structured_floor = {}
        self._output_ceiling = {}

    def redact(self, text: str) -> str:
        secrets = [(v, '[REDACTED]' if k.endswith('API_KEY') else '[API_ENDPOINT]')
                   for k, v in self.cfg.items() if v and k.endswith(('API_KEY', 'API_BASE_URL'))]
        for value, marker in sorted(secrets, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(value, marker)
        return text

    def create(self, messages, *, role='agent', max_output_tokens=4096, tools=None, schema=None, effort='medium', tool_choice=None, _retry=0, _structured=None, _shrink=0):
        if self.calls >= self.max_calls or (self.deadline and time.monotonic() >= self.deadline):
            raise ApiFailure('API call or episode time budget exhausted')
        if role == 'agent': bounded_timeout(300)
        route = role_config(self.cfg, role)
        model = route['MODEL']
        configured_effort = route['REASONING_EFFORT']
        effective_effort = effort if configured_effort == 'inherit' else (
            None if configured_effort == 'omit' else configured_effort)
        mode = route['API_MODE']
        structured = _structured or self._structured_floor.get(role) or route['STRUCTURED_OUTPUT_MODE']
        if structured not in ('json_schema', 'json_object', 'prompt'):
            raise ValueError('Unsupported structured output mode')
        schema_text = None if schema is None else (
            'Return only one JSON object conforming to this JSON Schema, with no prose outside it. Schema: '
            + json.dumps(schema))
        if mode == 'chat_completions':
            payload = chat_payload(messages, model=model, max_output_tokens=max_output_tokens,
                token_field=route['MAX_TOKENS_FIELD'], effort=effective_effort,
                tools=tools, schema=schema, tool_choice=tool_choice,
                structured_output_mode=structured)
            suffix = '/chat/completions'
        elif mode == 'responses':
            payload = {'model': model, 'input': messages, 'store': False, 'max_output_tokens': max_output_tokens}
            if effective_effort is not None:
                payload['reasoning'] = {'effort': effective_effort}
            if tools:
                payload.update(tools=tools, parallel_tool_calls=False)
                payload['include'] = ['reasoning.encrypted_content']
                if tool_choice: payload['tool_choice'] = tool_choice
            if schema_text is not None:
                if not isinstance(messages, list): raise ValueError('JSON mode requires message list')
                payload['input'] = [{'role': 'system', 'content': schema_text}] + messages
                if structured == 'json_object':
                    payload['text'] = {'format': {'type': 'json_object'}}
                elif structured == 'json_schema':
                    payload['text'] = {'format': {'type': 'json_schema', 'name': 'result', 'strict': True, 'schema': schema}}
            suffix = '/responses'
        else:
            raise ValueError('Unsupported API mode')
        options=json.loads(route['REQUEST_OPTIONS_JSON'])
        if not isinstance(options,dict) or set(options)-{'temperature','top_p','max_completion_tokens','max_output_tokens','max_tokens','seed','stop','extra_body'}:
            raise ValueError('Unsupported request options')
        extra=options.pop('extra_body',{})
        if not isinstance(extra,dict) or set(extra)-{'thinking'}:raise ValueError('Only thinking is supported in extra_body')
        payload.update(options);payload.update(extra)
        if role == 'agent' and schema is not None:
            floor = output_budget(effective_effort)
            ceiling = self._output_ceiling.get(role)
            for field in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'):
                current = payload.get(field)
                if isinstance(current, int) and not isinstance(current, bool):
                    sized = max(current, floor)
                    if ceiling:
                        sized = min(sized, ceiling)
                    payload[field] = sized
        headers = {'Content-Type': 'application/json'}
        if route['API_PROVIDER'] == 'azure':
            headers['api-key'] = route['API_KEY']
        elif route['API_PROVIDER'] == 'openai':
            headers['Authorization'] = 'Bearer ' + route['API_KEY']
        else:
            raise ValueError('Unsupported API provider')
        url = route['API_BASE_URL'].rstrip('/') + suffix
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
        metadata = public_routes(self.cfg)[role]
        started = time.monotonic()
        self.calls += 1
        timeout=float(route['HTTP_TIMEOUT_SECONDS'])
        if not __import__('math').isfinite(timeout) or timeout<=0:raise ValueError('Invalid role HTTP timeout')
        if role == 'agent' and schema is not None:
            sent = max((payload.get(field) or 0) for field in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'))
            timeout = generation_timeout(sent, timeout)
        timeout = min(timeout, max(1, self.deadline-started)) if self.deadline else timeout
        if role == 'agent': timeout = bounded_timeout(timeout)
        def retry_or_fail(message, transient):
            if self.log:
                self.log.parent.mkdir(parents=True,exist_ok=True)
                with self.log.open('a') as f:
                    f.write(json.dumps({'call':self.calls,'role':role,'model':model,'effort':effective_effort,
                      'api_route':metadata,
                      'seconds':round(time.monotonic()-started,3),'status':'error','usage':{},
                      'error':message,'retry':_retry,'usage_unknown':True})+'\n')
            if transient and _retry<1:
                # No tool action is executed until a complete response is returned.
                return self.create(messages,role=role,max_output_tokens=max_output_tokens,tools=tools,
                                   schema=schema,effort=effort,tool_choice=tool_choice,_retry=_retry+1,
                                   _structured=structured,_shrink=_shrink)
            raise ApiFailure(message) from None
        def fallback(reason):
            nxt = weaker_structured_mode(structured)
            if schema is None or not nxt:
                return None
            self._structured_floor[role] = nxt
            if self.log:
                self.log.parent.mkdir(parents=True, exist_ok=True)
                with self.log.open('a') as f:
                    f.write(self.redact(json.dumps({'call': self.calls, 'role': role, 'model': model,
                        'status': 'format_fallback', 'from': structured, 'to': nxt, 'reason': reason[:300]}))+'\n')
            return self.create(messages, role=role, max_output_tokens=max_output_tokens, tools=tools,
                               schema=schema, effort=effort, tool_choice=tool_choice, _retry=_retry,
                               _structured=nxt, _shrink=_shrink)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read()).get('error', {})
                code = self.redact(str(error.get('code', 'unknown')))[:80]
                message = self.redact(str(error.get('message', '')))[:600]
            except Exception:
                code, message = 'unparsed_error', 'response body withheld'
            if exc.code == 400 and token_limit_rejected(message) and schema is not None and _shrink < 4:
                sent = max((payload.get(field) or 0) for field in ('max_tokens', 'max_completion_tokens', 'max_output_tokens'))
                ceiling = token_range_ceiling(message) or lower_output_budget(sent)
                if ceiling and ceiling < sent:
                    self._output_ceiling[role] = min(self._output_ceiling.get(role, ceiling), ceiling)
                    if self.log:
                        self.log.parent.mkdir(parents=True, exist_ok=True)
                        with self.log.open('a') as f:
                            f.write(self.redact(json.dumps({'call': self.calls, 'role': role, 'model': model,
                                'status': 'token_ceiling', 'from': sent, 'to': self._output_ceiling[role]}))+'\n')
                    return self.create(messages, role=role, max_output_tokens=self._output_ceiling[role], tools=tools,
                                       schema=schema, effort=effort, tool_choice=tool_choice, _retry=_retry,
                                       _structured=structured, _shrink=_shrink+1)
            if exc.code == 400 and format_rejected(message):
                retried = fallback('http_400')
                if retried is not None:
                    return retried
            return retry_or_fail(f'HTTP {exc.code}; code={code}; {message}',exc.code==429 or exc.code>=500)
        except Exception as exc:
            return retry_or_fail(f'API transport failed: {type(exc).__name__}; details withheld',True)
        try:
            if not isinstance(result, dict):
                raise ValueError('Expected a JSON response object')
            if mode == 'chat_completions':
                result = chat_response(result)
        except (ValueError, KeyError, TypeError) as exc:
            return retry_or_fail('Invalid API response shape; no action executed (' + type(exc).__name__ + ')', False)
        usage = result.get('usage') or {}
        self.usage['input_tokens'] += usage.get('input_tokens', 0)
        self.usage['output_tokens'] += usage.get('output_tokens', 0)
        self.usage['cached_tokens'] += (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
        record = {'call': self.calls, 'role': role, 'model': model, 'effort': effective_effort,
                  'api_route': metadata,
                  'seconds': round(time.monotonic()-started, 3), 'usage': usage,
                  'status': result.get('status'), 'incomplete_details': result.get('incomplete_details')}
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open('a') as f:
                f.write(self.redact(json.dumps(record))+'\n')
        reason = (result.get('incomplete_details') or {}).get('reason') or result.get('chat_finish_reason')
        if schema is not None and structured == 'json_schema' and reason not in _LENGTH_REASONS:
            try:
                body = self.text(result)
            except (TypeError, KeyError, AttributeError):
                body = ''
            if body.strip() and not complete_json_object(body):
                retried = fallback('response_not_json')
                if retried is not None:
                    return retried
        return result

    @staticmethod
    def text(result):
        messages=[item for item in result.get('output',[]) if item.get('type')=='message']
        # Some deployments emit commentary AND final_answer, each containing JSON.
        # Joining them produces invalid JSON or could duplicate a command.
        finals=[item for item in messages if item.get('phase') in ('final_answer','final')]
        selected=(finals or messages)[-1:]  # One final message, never concatenate phases.
        return '\n'.join(c.get('text','') for item in selected for c in item.get('content',[])
                         if c.get('type')=='output_text')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--role', choices=('agent', 'simulator', 'judge'), default='agent')
    parser.add_argument('--check-config', action='store_true', help='Validate configuration without calling a model')
    args = parser.parse_args()
    try:
        client = Client(ROOT / 'runs' / 'api-smoke.jsonl', max_calls=1, env_file=args.env_file)
        if args.check_config:
            print(json.dumps({'configured': True, 'model_calls': 0, 'roles': public_routes(client.cfg)}, indent=2))
        else:
            result = client.create([{'role':'user','content':'Reply with exactly OK.'}], role=args.role,
                                   max_output_tokens=512, effort='low')
            print(json.dumps({'ok':client.text(result).strip() == 'OK', 'status':result.get('status'),
                              'role':args.role, 'api_route':public_routes(client.cfg)[args.role], 'usage':client.usage}))
    except (ApiFailure, ValueError) as e:
        print(str(e))
        raise SystemExit(1)
