"""Offline fault injection for completion caps, deadlines and action execution."""
from copy import deepcopy
import ast
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import urllib.error

import httpx
from openai import APIStatusError
from swe_persona.api import RoleAPI
from swe_persona.api_output import (IncompleteOutputError, OutputBudget, LENGTH_REASONS,
                                    coding_budget, provider_ceiling)
from swe_persona.api_recovery import APITransportExhausted

ROOT = Path(__file__).parents[1]
SCHEMA = {'type':'object','properties':{'ok':{'type':'boolean'}},
          'required':['ok'],'additionalProperties':False}
PACKAGE = 'fixture_deepswe_output_recovery'
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT/'src/swe_persona/vendor/deepswe/project/dspilot')]
sys.modules[PACKAGE] = package
ds = importlib.import_module(PACKAGE+'.api')
# The action backend's protocol depends on a selected dataset row at import.
# Supply only its static schema/file writer, leaving the actual API/backend intact.
protocol = ModuleType(PACKAGE+'.protocol')
schema_node = next(node for node in ast.parse((ROOT/'src/swe_persona/vendor/deepswe/project/dspilot/protocol.py').read_text()).body
                   if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='ACTION_SCHEMA' for target in node.targets))
protocol.ACTION_SCHEMA = ast.literal_eval(schema_node.value)
protocol.dump = lambda path,value: path.write_text(json.dumps(value))
sys.modules[protocol.__name__] = protocol
coding = importlib.import_module(PACKAGE+'.coding_json')


def sdk_error(status, message):
    return APIStatusError(message, response=httpx.Response(status,
        request=httpx.Request('POST','https://fixture.invalid')), body={'error':{'message':message}})


def sdk_response(text='{"ok":true}', reason='stop', api='chat_completions'):
    usage = NS(prompt_tokens=3,completion_tokens=7,input_tokens=3,output_tokens=7,
               completion_tokens_details=NS(reasoning_tokens=2),
               output_tokens_details=NS(reasoning_tokens=2))
    if api == 'responses':
        return NS(model='fixture',output_text=text,usage=usage,
            status='incomplete' if reason in LENGTH_REASONS else 'completed',
            incomplete_details=NS(reason=reason) if reason in LENGTH_REASONS else None)
    return NS(model='fixture',usage=usage,
        choices=[NS(finish_reason=reason,message=NS(content=text,reasoning_content='why'))])


class PublicOutputTests(unittest.TestCase):
    def api(self, outcomes, *, transport='chat_completions', options=None, role='agent', directory=None):
        create = Mock(side_effect=outcomes)
        client = NS(chat=NS(completions=NS(create=create)),responses=NS(create=create))
        client.with_options = Mock(return_value=client)
        config = {role:{'api_key_env':'FIXTURE_OUTPUT_KEY','base_url':'https://fixture.invalid/v1',
            'model':'fixture','api':transport,'timeout_sec':30,'structured_output_mode':'json_schema',
            'request_options':options if options is not None else {'reasoning_effort':'medium'}}}
        with patch.dict(os.environ,{'FIXTURE_OUTPUT_KEY':'dummy'}), patch('swe_persona.api.OpenAI',return_value=client):
            api = RoleAPI(config,role,directory)
        return api,create,client

    def tb3_call(self, api, role='agent'):
        from swe_persona.worker import install_tb3
        azure = ModuleType('azure_client')
        loop = ModuleType('requirement_loop')
        loop.POLICY = {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'row.json').write_text(json.dumps({'budget': {
                'max_user_turns': 24, 'max_simulator_decisions': 64}}))
            with patch.dict(sys.modules, {'azure_client': azure, 'requirement_loop': loop}), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch('swe_persona.worker.RoleAPI', return_value=api) as factory:
                install_tb3(root, {'configured': True}, root)
                result = azure.call(role, 'Return JSON', SCHEMA, root, timeout=30)
                self.assertEqual(factory.call_args.args[1], api.role)
                return result

    def test_tb3_agent_uses_effort_budgets_instead_of_fixed_cap(self):
        for effort, limit in ((None,65536),('none',8192),('low',16384),
                              ('medium',65536),('high',65536),('xhigh',131072),('max',131072)):
            with self.subTest(effort=effort):
                options = {} if effort is None else {'reasoning_effort': effort}
                api, create, _ = self.api([sdk_response()], options=options)
                self.assertEqual(self.tb3_call(api, 'coding_agent'), {'ok': True})
                self.assertEqual(create.call_args.kwargs['max_completion_tokens'], limit)

    def test_unstructured_agent_call_uses_the_same_effort_budget_and_growth(self):
        api,create,_=self.api([sdk_response('', 'length'),sdk_response()],
                              options={'reasoning_effort':'low'})
        self.assertEqual(api.raw([],schema=None)['text'],'{"ok":true}')
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],
                         [16384,32768])

    def test_unstructured_agent_never_executes_output_truncated_at_the_top_step(self):
        api,create,_=self.api([sdk_response('partial action', 'length')],
                              options={'reasoning_effort':'max'})
        with self.assertRaises(IncompleteOutputError):
            api.raw([],schema=None)
        self.assertEqual(create.call_args.kwargs['max_completion_tokens'],131072)

    def test_tb3_agent_recovers_truncation_with_shared_budget_growth(self):
        api, create, _ = self.api([sdk_response('', 'length'), sdk_response()])
        self.assertEqual(self.tb3_call(api), {'ok': True})
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],
                         [65536, 131072])

    def test_tb3_configured_agent_limit_remains_a_hard_ceiling(self):
        api, create, _ = self.api([sdk_response('', 'length'), sdk_response()],
                                  options={'reasoning_effort': 'high', 'max_tokens': 32768})
        self.assertEqual(self.tb3_call(api), {'ok': True})
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],
                         [32768, 32768])

    def test_tb3_non_agent_limits_are_preserved(self):
        for role in ('user_simulator', 'judge'):
            for options, expected in (({}, 16384), ({'max_output_tokens': 32768}, 32768)):
                with self.subTest(role=role, options=options):
                    api, create, _ = self.api([sdk_response(api='responses')],
                        transport='responses', role=role, options=options)
                    self.assertEqual(self.tb3_call(api, role), {'ok': True})
                    self.assertEqual(create.call_args.kwargs['max_output_tokens'], expected)

    def test_all_truncation_causes_recover_before_format_repair(self):
        for transport in ('chat_completions','responses'):
            for reason in LENGTH_REASONS:
                for partial in ('','{"ok":'):
                    with self.subTest(transport=transport,reason=reason,partial=partial):
                        api,create,_ = self.api([sdk_response(partial,reason,transport),sdk_response(api=transport)],transport=transport)
                        self.assertEqual(api.json([{'role':'user','content':'JSON'}],SCHEMA),{'ok':True})
                        field = 'max_output_tokens' if transport=='responses' else 'max_completion_tokens'
                        self.assertEqual([c.kwargs[field] for c in create.call_args_list],[65536,131072])
                        messages = 'input' if transport=='responses' else 'messages'
                        self.assertEqual(create.call_args_list[0].kwargs[messages],create.call_args_list[1].kwargs[messages])

    def test_only_one_increase_across_format_repairs(self):
        api,create,_ = self.api([sdk_response('', 'length'),sdk_response('invalid'),sdk_response()])
        self.assertEqual(api.json([{'role':'user','content':'JSON'}],SCHEMA),{'ok':True})
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],[65536,131072,131072])
        self.assertIn('schema mismatch',create.call_args.kwargs['messages'][-1]['content'])

    def test_empty_or_malformed_output_without_cutoff_does_not_grow(self):
        for text in ('','bad'):
            with self.subTest(text=text):
                api,create,_=self.api([sdk_response(text),sdk_response()])
                self.assertEqual(api.json([],SCHEMA),{'ok':True})
                self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],[65536,65536])

    def test_repeated_cutoffs_have_bounded_increase_and_format_repairs(self):
        api,create,_ = self.api([sdk_response('', 'length') for _ in range(4)])
        with self.assertRaises(json.JSONDecodeError):api.json([],SCHEMA)
        self.assertEqual(create.call_count,4)
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],[65536,131072,131072,131072])

    def test_configured_and_caller_limits_are_hard_ceilings(self):
        for transport,field in (('chat_completions','max_completion_tokens'),('responses','max_output_tokens')):
            with self.subTest(transport=transport):
                options={'reasoning_effort':'medium',field:10000}
                original=deepcopy(options)
                api,create,client=self.api([sdk_response('', 'length',transport)],transport=transport,options=options)
                with self.assertRaises(IncompleteOutputError):api.raw([],SCHEMA)
                self.assertEqual(create.call_count,1)
                self.assertEqual(create.call_args.kwargs[field],10000)
                self.assertLessEqual(client.with_options.call_args.kwargs['timeout'],30)
                self.assertEqual(options,original)
        api,create,_=self.api([sdk_response('', 'length')])
        with self.assertRaises(IncompleteOutputError):api.raw([],SCHEMA,max_tokens=5000)
        self.assertEqual(create.call_count,1)
        self.assertEqual(create.call_args.kwargs['max_completion_tokens'],5000)

    def test_configuration_overrides_profile_limit_without_automatic_increase(self):
        api,create,_=self.api([sdk_response()],options={'max_tokens':12000})
        api.raw([],SCHEMA,max_tokens=4000)
        self.assertEqual(create.call_args.kwargs['max_completion_tokens'],12000)
        self.assertNotIn('max_tokens',create.call_args.kwargs)

    def test_provider_ceiling_is_remembered_and_configuration_is_immutable(self):
        message='max_completion_tokens must be in range [1, 32768]'
        api,create,_=self.api([sdk_error(400,message),sdk_response(),sdk_response()])
        original=deepcopy(api.config)
        api.raw([],SCHEMA);api.raw([],SCHEMA)
        self.assertEqual([c.kwargs['max_completion_tokens'] for c in create.call_args_list],[65536,32768,32768])
        self.assertEqual(api.config,original)

    def test_auth_schema_and_unrelated_400_are_not_retried_or_downgraded(self):
        for status,message in ((401,'unauthorized'),(400,'response_format json_schema invalid'),
                               (400,'max_completion_tokens must be an integer')):
            with self.subTest(status=status,message=message):
                api,create,_=self.api([sdk_error(status,message)])
                with self.assertRaises(APIStatusError):api.raw([],SCHEMA)
                self.assertEqual(create.call_count,1)
                self.assertEqual(create.call_args.kwargs['response_format']['type'],'json_schema')

    def test_default_simulator_keeps_responses_conversion_and_does_not_expand(self):
        api,create,_=self.api([sdk_response('', 'max_output_tokens','responses')],transport='responses',
            role='user_simulator',options={'reasoning_effort':'medium','max_output_tokens':32768})
        with self.assertRaises(IncompleteOutputError):api.raw([],SCHEMA)
        self.assertEqual(create.call_count,1)
        self.assertEqual(create.call_args.kwargs['reasoning'],{'effort':'medium'})
        self.assertNotIn('reasoning_effort',create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs['max_output_tokens'],32768)

    def test_complete_looking_json_with_cutoff_flag_is_not_accepted(self):
        api,create,_=self.api([sdk_response(reason='length') for _ in range(3)],options={'max_completion_tokens':10000})
        with self.assertRaises(json.JSONDecodeError):api.json([],SCHEMA)
        self.assertEqual(create.call_count,3)
        api,create,_=self.api([sdk_response(reason='length')],options={'max_completion_tokens':10000})
        with self.assertRaises(IncompleteOutputError):api.raw([],SCHEMA)
        self.assertEqual(create.call_count,1)

    def test_transport_length_and_format_repair_share_deadline(self):
        clock=NS(now=0.0);timeouts=[];calls=[]
        outcomes=[TimeoutError(),sdk_response('', 'length'),sdk_response('bad'),sdk_response()]
        def create(**kwargs):
            calls.append(kwargs);clock.now+=min(10,timeouts[-1])
            response=outcomes.pop(0)
            if isinstance(response,Exception):raise response
            return response
        api,operation,client=self.api([])
        operation.side_effect=create
        client.with_options.side_effect=lambda **opts:(timeouts.append(opts['timeout']) or client)
        with patch('swe_persona.api.time.monotonic',side_effect=lambda:clock.now), \
             patch('swe_persona.api_recovery.time.sleep',side_effect=lambda seconds:setattr(clock,'now',clock.now+seconds)):
            self.assertEqual(api.json([],SCHEMA,timeout=35),{'ok':True})
        self.assertEqual(timeouts,[30,24,14,4])
        self.assertEqual(clock.now,35)
        self.assertEqual([c['max_completion_tokens'] for c in calls],[65536,65536,131072,131072])

    def test_provider_adjustment_stops_at_shared_deadline(self):
        clock=NS(now=0.0)
        def fail(**kwargs):
            clock.now+=5
            raise sdk_error(400,'max_completion_tokens maximum 32768')
        api,create,_=self.api([]);create.side_effect=fail
        with patch('swe_persona.api.time.monotonic',side_effect=lambda:clock.now):
            with self.assertRaises(APITransportExhausted):api.raw([],SCHEMA,timeout=5)
        self.assertEqual(create.call_count,1)

    def test_each_generated_attempt_is_audited_and_usage_is_summed(self):
        with tempfile.TemporaryDirectory() as temp:
            api,_,_=self.api([sdk_response('', 'length'),sdk_response()],directory=Path(temp))
            result=api.raw([],SCHEMA)
            self.assertEqual(result['output_tokens'],14)
            records=[json.loads(line) for line in (Path(temp)/'api-calls.jsonl').read_text().splitlines()]
            self.assertEqual([r['requested_max_tokens'] for r in records],[65536,131072])
            self.assertEqual([r['output_tokens'] for r in records],[7,7])
            self.assertEqual([r['effort'] for r in records],['medium','medium'])
            self.assertEqual([r['finish_reason'] for r in records],['length','stop'])
            self.assertEqual([r['reasoning_tokens'] for r in records],[2,2])
            self.assertEqual([r['visible_output_chars'] for r in records],[0,len('{"ok":true}')])
            audit=json.loads((Path(temp)/'transport-recovery.jsonl').read_text())
            self.assertEqual(audit['status'],'length_retry')
            self.assertFalse(audit['action_executed'])


class BudgetTests(unittest.TestCase):
    def test_effort_defaults_and_top_step(self):
        for effort,limit in (('none',8192),('minimal',8192),('low',16384),('medium',65536),
                             ('high',65536),('xhigh',131072),('max',131072)):
            with self.subTest(effort=effort):
                budget=OutputBudget.resolve(adaptive=True,effort=effort)
                self.assertEqual(budget.current,limit)
                self.assertEqual(budget.grow('length'),effort not in ('xhigh','max'))
                self.assertFalse(budget.grow('max_output_tokens'))

    def test_shared_coding_budget_keeps_explicit_configuration_authoritative(self):
        self.assertEqual(coding_budget({'reasoning_effort':'max'}).current,131072)
        budget=coding_budget({'reasoning_effort':'max','max_tokens':24000},requested=4096)
        self.assertEqual(budget.current,24000)
        self.assertTrue(budget.explicit)
        self.assertFalse(budget.grow('length'))

    def test_provider_range_parsing_and_bounded_shrinking(self):
        for message in ('max_tokens range [1, 2048]','max_tokens must be between 1 and 2048',
                        'max_tokens cannot exceed 2048','max_tokens must be <= 2048'):
            self.assertEqual(provider_ceiling(400,message,10000),2048)
        self.assertIsNone(provider_ceiling(400,'json_schema maximum 10',65536))
        self.assertIsNone(provider_ceiling(401,'max_tokens maximum 10',65536))
        budget=OutputBudget.resolve(adaptive=True)
        for _ in range(4):self.assertTrue(budget.shrink(400,'max_tokens too large'))
        self.assertFalse(budget.shrink(400,'max_tokens too large'))


class DeepOutputTests(unittest.TestCase):
    def client(self,*,transport='responses',options=None,deadline=None,log=None):
        config={'MODEL':'fixture','API_BASE_URL':'https://fixture.invalid/v1','API_KEY':'dummy',
            'API_PROVIDER':'openai','API_MODE':transport,'REASONING_EFFORT':'medium',
            'STRUCTURED_OUTPUT_MODE':'json_schema','HTTP_TIMEOUT_SECONDS':'30',
            'REQUEST_OPTIONS_JSON':json.dumps(options or {})}
        with patch.object(ds,'config',return_value=config):return ds.Client(log,max_calls=20,deadline=deadline)

    def response(self,text='{"ok":true}',reason='stop',transport='responses'):
        if transport=='chat_completions':
            data={'id':'fixture','choices':[{'finish_reason':reason,'message':{'content':text}}],
                  'usage':{'prompt_tokens':3,'completion_tokens':7,
                           'completion_tokens_details':{'reasoning_tokens':2}}}
        else:
            data={'status':'incomplete' if reason in LENGTH_REASONS else 'completed',
                'incomplete_details':{'reason':reason} if reason in LENGTH_REASONS else None,
                'output':[{'type':'message','content':[{'type':'output_text','text':text}]}],
                'usage':{'input_tokens':3,'output_tokens':7,
                         'output_tokens_details':{'reasoning_tokens':2}}}
        return io.BytesIO(json.dumps(data).encode())

    def request_mock(self,outcomes,records):
        def request(req,timeout):
            records.append((json.loads(req.data),timeout))
            response=outcomes.pop(0)
            if isinstance(response,Exception):raise response
            return response
        return request

    def test_both_transports_and_all_cutoff_causes(self):
        for transport in ('responses','chat_completions'):
            for reason in LENGTH_REASONS:
                with self.subTest(transport=transport,reason=reason):
                    client=self.client(transport=transport);records=[]
                    outcomes=[self.response('',reason,transport),self.response(transport=transport)]
                    with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
                         patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock(outcomes,records)):
                        response=client.create([],schema=SCHEMA)
                    field='max_output_tokens' if transport=='responses' else 'max_completion_tokens'
                    self.assertEqual([r[0][field] for r in records],[65536,131072])
                    self.assertEqual(response['status'],'completed')
                    self.assertEqual(client.calls,2)
                    self.assertEqual(client.usage['output_tokens'],14)

    def test_deepswe_audit_records_common_agent_budget_fields(self):
        for transport in ('responses','chat_completions'):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as temp:
                log=Path(temp)/'api.jsonl';client=self.client(transport=transport,log=log);records=[]
                with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
                     patch.object(ds.urllib.request,'urlopen',
                                  side_effect=self.request_mock([self.response(transport=transport)],records)):
                    client.create([],schema=SCHEMA,effort='max')
                audit=json.loads(log.read_text())
                self.assertEqual(audit['requested_max_tokens'],65536)
                self.assertEqual(audit['effort'],'medium')
                self.assertEqual(audit['finish_reason'],'completed' if transport=='responses' else 'stop')
                self.assertEqual(audit['reasoning_tokens'],2)
                self.assertEqual(audit['visible_output_chars'],len('{"ok":true}'))

    def test_provider_downshift_is_cached_and_keeps_one_token_field(self):
        client=self.client(transport='chat_completions',options={'max_tokens':10000});records=[]
        error=urllib.error.HTTPError('https://fixture.invalid',400,'range',{},io.BytesIO(
            json.dumps({'error':{'message':'max_completion_tokens range [1, 4096]'}}).encode()))
        outcomes=[error,self.response(transport='chat_completions'),self.response(transport='chat_completions')]
        with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock(outcomes,records)):
            client.create([],schema=SCHEMA);client.create([],schema=SCHEMA)
        self.assertEqual([r[0]['max_completion_tokens'] for r in records],[10000,4096,4096])
        self.assertTrue(all('max_tokens' not in r[0] for r in records))
        self.assertEqual(json.loads(client.cfg['REQUEST_OPTIONS_JSON']),{'max_tokens':10000})

    def test_downshift_transport_retry_and_growth_share_deadline_and_call_budget(self):
        clock=NS(now=0.0);client=self.client(deadline=15);records=[]
        error=urllib.error.HTTPError('https://fixture.invalid',400,'range',{},io.BytesIO(
            b'{"error":{"message":"max_output_tokens maximum 32768"}}'))
        responses=[TimeoutError(),error,self.response('', 'max_output_tokens')]
        def request(req,timeout):
            records.append((json.loads(req.data),timeout));clock.now+=min(5,timeout)
            response=responses.pop(0)
            if isinstance(response,Exception):raise response
            return response
        with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.time,'monotonic',side_effect=lambda:clock.now), \
             patch('swe_persona.api_recovery.time.sleep',side_effect=lambda seconds:setattr(clock,'now',clock.now+seconds)), \
             patch.object(ds.urllib.request,'urlopen',side_effect=request):
            response=client.create([],schema=SCHEMA)
        self.assertEqual([r[1] for r in records],[15,9,4])
        self.assertEqual([r[0]['max_output_tokens'] for r in records],[65536,65536,32768])
        self.assertEqual(response['status'],'incomplete')
        self.assertEqual(client.calls,3)
        self.assertEqual(clock.now,15)

    def test_call_limit_prevents_extra_generation(self):
        client=self.client();client.max_calls=1;records=[]
        with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock([self.response('', 'length')],records)):
            with self.assertRaises(ds.ApiFailure):client.create([],schema=SCHEMA)
        self.assertEqual(len(records),1)
        self.assertEqual(client.calls,1)

    def test_coding_backend_preserves_budget_and_has_no_length_protocol_rejection(self):
        action=json.dumps({'action':'reply','content':'ok','phase':'working'})
        client=self.client();records=[];outcomes=[self.response('', 'max_output_tokens'),self.response('bad'),self.response(action)]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock(outcomes,records)):
            rejected=[];backend=coding.JsonActBackend(client,Path(temp),rejected)
            result,identity,_=backend.next_action(1)
            self.assertEqual(result['action'],'reply')
            self.assertEqual(backend.pending,identity)
            self.assertEqual(len(rejected),1)
            self.assertFalse(rejected[0]['executed'])
            self.assertEqual([r[0]['max_output_tokens'] for r in records],[65536,131072,131072])
            self.assertEqual(len(list(Path(temp).glob('response-*.json'))),1)

    def test_configured_limit_never_grows_and_simulator_never_adapts(self):
        for role,options,requested,expected in (('agent',{'max_output_tokens':9000},None,9000),
                                                ('agent',{},7000,7000),('simulator',{},None,4096)):
            with self.subTest(role=role):
                client=self.client(options=options);records=[]
                with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
                     patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock([self.response('', 'length')],records)):
                    client.create([],schema=SCHEMA,role=role,max_output_tokens=requested)
                self.assertEqual(len(records),1)
                self.assertEqual(records[0][0]['max_output_tokens'],expected)

    def test_native_tool_transport_does_not_get_json_defaults(self):
        client=self.client();records=[]
        with patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.urllib.request,'urlopen',side_effect=self.request_mock([self.response()],records)):
            client.create([],max_output_tokens=3000)
        self.assertEqual(records[0][0]['max_output_tokens'],3000)
        self.assertNotIn('text',records[0][0])

    def test_backend_format_repair_shares_absolute_deadline(self):
        clock=NS(now=0.0);client=self.client(deadline=10);records=[]
        action=json.dumps({'action':'reply','content':'ok','phase':'working'})
        responses=[self.response('bad'),self.response(action)]
        def request(req,timeout):
            records.append(timeout);clock.now+=min(6,timeout)
            return responses.pop(0)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(ds,'bounded_timeout',side_effect=lambda seconds:seconds), \
             patch.object(ds.time,'monotonic',side_effect=lambda:clock.now), \
             patch.object(ds.urllib.request,'urlopen',side_effect=request):
            backend=coding.JsonActBackend(client,Path(temp),[])
            backend.next_action(1)
        self.assertEqual(records,[10,4])
        self.assertEqual(clock.now,10)
