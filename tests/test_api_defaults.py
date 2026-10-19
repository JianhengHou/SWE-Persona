"""Simulator defaults and Responses request routing."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from swe_persona.api import DEFAULT_USER_SIMULATOR_CONFIG, RoleAPI, chat_token_options, load_config, responses_options


class APIDefaultTests(unittest.TestCase):
    def test_simulator_default_does_not_follow_coding_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            for model in ('coding-model-a', 'coding-model-b'):
                path.write_text(json.dumps({'agent': dict(DEFAULT_USER_SIMULATOR_CONFIG, model=model,
                                                         api_key_env='AGENT_KEY')}))
                config=load_config(path)
                self.assertEqual(config['agent']['model'], model)
                self.assertEqual(config['user_simulator']['model'], 'gpt-6-luna')
                self.assertEqual(config['user_simulator']['request_options']['reasoning_effort'], 'medium')
                self.assertEqual(config['judge'], config['user_simulator'])
                config['judge']['request_options']['reasoning_effort']='high'
                self.assertEqual(config['user_simulator']['request_options']['reasoning_effort'], 'medium')

    def test_explicit_simulator_override_and_conflicting_options(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            path.write_text(json.dumps({'agent': DEFAULT_USER_SIMULATOR_CONFIG,
                                        'user_simulator': dict(DEFAULT_USER_SIMULATOR_CONFIG, model='explicit-model')}))
            self.assertEqual(load_config(path)['user_simulator']['model'], 'explicit-model')
        with self.assertRaises(ValueError):
            responses_options({'request_options': {'reasoning_effort':'medium','reasoning':{'effort':'high'}}})
        with self.assertRaises(ValueError):
            responses_options({'request_options': {'max_output_tokens':12,'max_tokens':13}})

    def test_zai_thinking_is_forwarded_in_extra_body(self):
        options=chat_token_options({
            'max_tokens_field':'max_completion_tokens',
            'request_options':{'reasoning_effort':'max',
                               'thinking':{'type':'enabled','clear_thinking':False}}})
        self.assertEqual(options['extra_body']['thinking'],{'type':'enabled','clear_thinking':False})
        self.assertNotIn('thinking',options)
        self.assertEqual(options['reasoning_effort'],'max')

    def test_responses_uses_medium_and_configured_limit_with_schema_validation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'SWE_PERSONA_SIMULATOR_API_KEY':'fixture'}):
            create=Mock(side_effect=[
                SimpleNamespace(output_text='{"wrong":true}',model='gpt-6-luna',status='completed',usage=None),
                SimpleNamespace(output_text='{"ready":true}',model='gpt-6-luna',status='completed',usage=None)])
            client=SimpleNamespace(responses=SimpleNamespace(create=create))
            client.with_options=lambda **options:client
            config={'user_simulator':deepcopy(DEFAULT_USER_SIMULATOR_CONFIG)}
            schema={'type':'object','properties':{'ready':{'type':'boolean'}},'required':['ready'],'additionalProperties':False}
            with patch('swe_persona.api.OpenAI',return_value=client):
                value=RoleAPI(config,'user_simulator',Path(temp)).json([{'role':'user','content':'ready'}],schema)
            self.assertEqual(value,{'ready':True})
            self.assertEqual(create.call_count,2)
            for call in create.call_args_list:
                self.assertEqual(call.kwargs['reasoning'],{'effort':'medium'})
                self.assertNotIn('reasoning_effort',call.kwargs)
                self.assertEqual(call.kwargs['max_output_tokens'],32768)
                self.assertEqual(call.kwargs['text']['format']['type'],'json_schema')
            self.assertIn('schema mismatch',create.call_args.kwargs['input'][-1]['content'])
            audit=json.loads((Path(temp)/'api-calls.jsonl').read_text().splitlines()[-1])
            self.assertEqual(audit['requested_max_tokens'],32768)
