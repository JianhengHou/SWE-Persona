"""Translate the runner's Responses history to/from Chat Completions.

Only text messages and function tools are supported. Unsupported input is
rejected before a request instead of silently removing executable context.
"""
import copy
import json


def _text(content):
    if content is None or isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get('type') not in ('input_text', 'output_text', 'text'):
            raise ValueError('Chat adapter supports text content only')
        parts.append(part['text'])
    return '\n'.join(parts)


def chat_messages(items):
    if isinstance(items, str):
        return [{'role': 'user', 'content': items}]
    messages = []
    for item in items:
        kind = item.get('type', 'message')
        if kind == 'reasoning':
            # Encrypted Responses reasoning is provider-specific, not a tool
            # result or a user instruction. It cannot be sent to Chat APIs.
            continue
        if kind == 'function_call':
            call = {'id': item['call_id'], 'type': 'function',
                    'function': {'name': item['name'], 'arguments': item['arguments']}}
            if messages and messages[-1]['role'] == 'assistant' and messages[-1].get('tool_calls'):
                messages[-1]['tool_calls'].append(call)
            else:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [call]})
        elif kind == 'function_call_output':
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': _text(item['output'])})
        elif kind == 'message':
            role = item.get('role')
            if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
                raise ValueError('Unsupported chat message role')
            msg = {'role': role, 'content': _text(item.get('content'))}
            for key in ('tool_calls', 'tool_call_id', 'name'):
                if key in item:
                    msg[key] = copy.deepcopy(item[key])
            messages.append(msg)
        else:
            raise ValueError('Unsupported Responses item for Chat Completions')
    return messages


def chat_payload(messages, *, model, max_output_tokens, token_field, effort, tools, schema, tool_choice, structured_output_mode="json_schema"):
    payload = {'model': model, 'messages': chat_messages(messages), token_field: max_output_tokens}
    if effort is not None:
        payload['reasoning_effort'] = effort
    if tools:
        functions = []
        for tool in tools:
            if tool.get('type') != 'function':
                raise ValueError('Chat adapter supports function tools only')
            functions.append({'type': 'function', 'function': {
                key: copy.deepcopy(tool[key]) for key in ('name', 'description', 'parameters', 'strict') if key in tool}})
        payload.update(tools=functions, parallel_tool_calls=False)
        if tool_choice:
            payload['tool_choice'] = ({'type': 'function', 'function': {'name': tool_choice['name']}}
                                     if isinstance(tool_choice, dict) else tool_choice)
    if schema is not None:
        if structured_output_mode in ('json_object','prompt'):
            if structured_output_mode == 'json_object':
                payload['response_format'] = {'type': 'json_object'}
            instruction = ('\nOutput transport: return ONLY one JSON object conforming to this schema. '
                'Natural-language role content belongs inside the appropriate string fields, never '
                'before or after the JSON envelope. Do not use Markdown fences. Schema: '
                + json.dumps(schema, ensure_ascii=False))
            indices = [i for i,m in enumerate(payload['messages']) if m.get('role') == 'system']
            if indices:
                payload['messages'][indices[-1]]['content'] += instruction
            else:
                payload['messages'].insert(0, {'role':'system','content':instruction})
        elif structured_output_mode == 'json_schema':
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'result', 'strict': True, 'schema': schema}}
        else:
            raise ValueError('Unsupported structured output mode')
    return payload


def chat_response(raw):
    choices = raw.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Expected one Chat Completions choice')
    choice = choices[0]
    msg = choice['message']
    finish = choice.get('finish_reason')
    status = 'completed' if finish in ('stop', 'tool_calls') else 'incomplete'
    output = []
    if msg.get('refusal'):
        status = 'failed'
        output.append({'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'refusal', 'refusal': msg['refusal']}]})
    elif msg.get('content') is not None:
        if not isinstance(msg['content'], str):
            raise ValueError('Expected text in Chat Completions response')
        output.append({'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'output_text', 'text': msg['content']}]})
    for call in msg.get('tool_calls') or []:
        if call.get('type') != 'function':
            raise ValueError('Unsupported Chat Completions tool response')
        output.append({'type': 'function_call', 'call_id': call['id'],
                       'name': call['function']['name'], 'arguments': call['function']['arguments'],
                       'status': 'completed' if status == 'completed' else 'incomplete'})
    usage = raw.get('usage') or {}
    return {'id': raw.get('id'), 'model': raw.get('model'), 'status': status,
            'output': output, 'chat_finish_reason': finish,
            'incomplete_details': None if status == 'completed' else {'reason': finish or 'missing_finish_reason'},
            'usage': {'input_tokens': usage.get('prompt_tokens', 0),
                      'output_tokens': usage.get('completion_tokens', 0),
                      'output_tokens_details': {
                          'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')},
                      'input_tokens_details': {'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)}}}


def structured_object(raw):
    """Decode one object, accepting an audited prose prefix from JSON-mode APIs.

    Only a single terminal JSON object is accepted. Never execute or deliver the
    prefix; callers validate the original schema and retain the full raw response.
    Ambiguous objects, braces in the prefix, and any trailing content are rejected.
    """
    text=raw.strip()
    if text.startswith('```') and text.endswith('```') and '\n' in text:
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    try:
        value=json.loads(text)
    except json.JSONDecodeError:
        start=text.find('{')
        if start<=0 or any(c in text[:start] for c in '{}[]'):
            raise
        value,end=json.JSONDecoder().raw_decode(text[start:])
        if text[start+end:].strip():
            raise ValueError('Trailing content or multiple JSON values')
    if not isinstance(value,dict):
        raise ValueError('Expected a JSON object')
    return value
