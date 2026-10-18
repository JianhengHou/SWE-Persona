"""Offline action-protocol tests, including recovery and execution boundaries."""
import ast
import asyncio
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from jsonschema import Draft202012Validator, ValidationError
from swe_persona.agent_output import action_envelope

ROOT = Path(__file__).parents[1]
VENDOR = ROOT/'src/swe_persona/vendor'


def file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


deep_package = package('fixture_deep_agent_envelope', VENDOR/'deepswe/project/dspilot')
protocol = types.ModuleType(deep_package.__name__+'.protocol')
# The schema is an actual static declaration. Avoid selecting a dataset task
# just to import this task-independent transport and parser.
tree = ast.parse((Path(deep_package.__path__[0])/'protocol.py').read_text())
schema_node = next(node for node in tree.body if isinstance(node, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id=='ACTION_SCHEMA' for t in node.targets))
protocol.ACTION_SCHEMA = ast.literal_eval(schema_node.value)
protocol.dump = lambda path,value: path.write_text(json.dumps(value))
sys.modules[protocol.__name__] = protocol
deep = importlib.import_module(deep_package.__name__+'.coding_json')
tp = package('fixture_together_agent_envelope', VENDOR/'swe_together/swe_together_pilot/persona_runtime')
together = importlib.import_module(tp.__name__+'.public_run')
with patch.dict(sys.modules), patch.object(sys, 'path', [str(VENDOR/'tb3/runtime'), *sys.path]):
    tb3 = file_module('fixture_tb3_agent_envelope', VENDOR/'tb3/runtime/terminal_agent.py')


def examples(prompt):
    block = prompt.split('## Output example\n',1)[1].split('\n\n## Role\n',1)[0]
    return [json.loads(line) for line in block.splitlines() if line.startswith('{')]


PROFILES = ((deep.AGENT_PROTOCOL, protocol.ACTION_SCHEMA),
            (together.AGENT_PROMPT, together.AGENT_SCHEMA),
            (tb3.AGENT_ENVELOPE, tb3.ACTION))


class AgentEnvelopeTests(unittest.TestCase):
    def test_examples_are_schema_valid_complete_and_have_no_task_specific_facts(self):
        for prompt, schema in PROFILES:
            with self.subTest(fields=schema['required']):
                self.assertTrue(prompt.startswith('## Response envelope'))
                self.assertEqual(prompt.count('## Response envelope'),1)
                self.assertIn('Key order is unrestricted',prompt)
                self.assertIn('an example does not require an additional approval step',prompt)
                for text in ('routing.py','auto_head','auto_options','X-Auto','until your next message','until you say so'):
                    self.assertNotIn(text,prompt)
                for value in examples(prompt):
                    Draft202012Validator(schema).validate(value)
                    self.assertEqual(set(value),set(schema['required']))
                    Draft202012Validator(schema).validate(dict(reversed(list(value.items()))))

    def test_incomplete_or_invalid_examples_are_rejected_before_prompt_use(self):
        schema = together.AGENT_SCHEMA
        fields = {key:key for key in schema['properties']}
        for value in ({'action':'message','content':'text'},
                      {'action':'message','content':'text','completion_claim':'false'},
                      {'action':'message','content':'text','completion_claim':False,'extra':'hidden'}):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                action_envelope(schema,[value],fields)

    def test_json_escaping_roundtrips_quotes_newlines_and_backslashes(self):
        value = {'action':'message','content':'Line one\n"quoted" \\ path','completion_claim':False}
        schema = together.AGENT_SCHEMA
        prompt = action_envelope(schema,[value],{key:key for key in schema['properties']})
        self.assertEqual(examples(prompt),[value])
        self.assertIn(r'a newline is `\n`, a quote is `\"`, and a backslash is `\\`',prompt)

    def test_existing_permissions_and_completion_review_are_explicit(self):
        self.assertIn('Task instructions and permissions come from the user',deep.AGENT_PROTOCOL)
        self.assertIn('waits for the user',deep.AGENT_PROTOCOL)
        self.assertIn('Follow the user\'s scope and authorization boundaries',together.AGENT_PROMPT)
        self.assertIn('final benchmark acceptance',together.AGENT_PROMPT)
        self.assertIn('does not declare benchmark success',tb3.AGENT_ENVELOPE)
        self.assertIn('without executing it',tb3.AGENT_ENVELOPE)
        self.assertIn('actual path supplied or discovered in this task',tb3.AGENT_ENVELOPE)


class DeepExecutionTests(unittest.TestCase):
    def client(self, results):
        client = Mock()
        client.create.side_effect = [copy.deepcopy(value) for value in results]
        client.text.side_effect = lambda response: response['text']
        return client

    def response(self, action):
        return {'status':'completed','output':[],'text':json.dumps(action)}

    def test_real_parser_accepts_every_action_phase_and_reordered_keys(self):
        for action in ('shell','validate','reply'):
            for phase in ('planning','working','question','done'):
                value = {'content':'' if action=='validate' else 'Current task text.', 'phase':phase,'action':action}
                client = self.client([])
                self.assertEqual(deep.validate(self.response(value),client),value)
        self.assertEqual(protocol.ACTION_SCHEMA['required'],['action','content','phase'])

    def test_rejected_response_cannot_become_pending_or_execute(self):
        value = {'action':'shell','phase':'working','content':'pwd'}
        invalid = (self.response({**value,'extra':'x'}),
                   self.response({k:v for k,v in value.items() if k!='phase'}),
                   {**self.response(value),'text':'Explanation\n'+json.dumps(value)},
                   {**self.response(value),'output':[{'type':'function_call'}]})
        for bad in invalid:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                client = self.client([bad,bad,bad])
                rejected = []
                backend = deep.JsonActBackend(client,Path(directory),rejected)
                with self.assertRaisesRegex(ValueError,'bounded protocol recovery'):
                    backend.next_action(1)
                self.assertEqual(client.create.call_count,3)
                self.assertIsNone(backend.pending)
                self.assertTrue(all(not item['executed'] for item in rejected))
                self.assertFalse((Path(directory)/'response-001.json').exists())

    def test_nul_in_shell_content_gets_protocol_repair_instead_of_reaching_docker(self):
        # As observed: a model emitted \u0000 where '&' belonged, and docker exec raised "embedded null byte".
        bad = {'action':'shell','phase':'working','content':"sed -i 's/x/atomic.AddInt32(\x00f.n, -1)/' mux.go"}
        good = {**bad,'content':"sed -i 's/x/atomic.AddInt32(\\&f.n, -1)/' mux.go"}
        with tempfile.TemporaryDirectory() as directory:
            client = self.client([self.response(bad),self.response(good)])
            rejected = []
            backend = deep.JsonActBackend(client,Path(directory),rejected)
            action,_,_ = backend.next_action(1)
            self.assertEqual(action,good)
            self.assertEqual([item['executed'] for item in rejected],[False])
            self.assertIn('U+0000',rejected[0]['error'])
            self.assertIn('U+0000',backend.history[-2]['content'])

    def test_format_repair_preserves_schema_budget_deadline_and_user_authority(self):
        value = {'phase':'planning','content':'Here is my plan.','action':'reply'}
        client = self.client([{'status':'completed','output':[],'text':'Invalid prose'},self.response(value)])
        with tempfile.TemporaryDirectory() as directory:
            backend = deep.JsonActBackend(client,Path(directory),[])
            authority = 'Review only. Do not edit or commit repository files.'
            backend.deliver_user(authority)
            action, identity, _ = backend.next_action(1)
            self.assertEqual(action,value)
            self.assertEqual(identity,backend.pending)
            first, second = client.create.call_args_list
            self.assertIs(first.kwargs['_budget'],second.kwargs['_budget'])
            self.assertIs(first.kwargs['_deadline'],second.kwargs['_deadline'])
            self.assertIs(second.kwargs['schema'],protocol.ACTION_SCHEMA)
            self.assertNotIn('max_output_tokens',second.kwargs)
            self.assertIn({'role':'user','content':authority},backend.history)
            repair = backend.history[-2]['content']
            self.assertIn('Key order is unrestricted',repair)
            self.assertIn('no action was executed',repair)
            self.assertIn('not a user request',repair)
            with self.assertRaisesRegex(ValueError,'no observation'):backend.next_action(2)
            backend.observe(identity,{'delivered':True})
            self.assertIsNone(backend.pending)


class TerminalExecutionTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, directory):
        agent = tb3.PersonaTerminalAgent.__new__(tb3.PersonaTerminalAgent)
        agent.arm = 'single_turn'; agent.limit = 60; agent.binding = None
        agent.private_dir = Path(directory); agent.expired = False; agent.paused = False
        agent.end = None; agent.cutoff_at = None; agent.requirement_loop = None
        agent.actions_since_user = 0; agent.events = []; agent.start = 0
        agent.event = Mock(); agent.save = Mock(); agent.pause = AsyncMock()
        agent.checkpoint = AsyncMock()
        async def watch(): await asyncio.Future()
        agent.watch = watch
        return agent

    def action(self, kind, command='', message=''):
        return {'action':kind,'command':command,'message':message,'image_path':''}

    async def test_actual_run_prepends_protocol_and_proposed_command_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.fixture(directory)
            env = types.SimpleNamespace(exec=AsyncMock())
            actions = [self.action('reply',command='printf proposed',message='May I run this command?'),
                       self.action('ready',message='Please review the work.')]
            context = types.SimpleNamespace(metadata=None)
            with patch.object(tb3,'call',side_effect=actions) as call:
                await agent.run('Current user instructions.',env,context)
            env.exec.assert_not_awaited()
            prompt = call.call_args_list[0].args[1]
            self.assertTrue(prompt.startswith(tb3.AGENT_ENVELOPE))
            self.assertIn('when the user requires confirmation',prompt)
            data = json.loads(prompt.split('\nINPUT JSON:\n',1)[1])
            self.assertEqual(data['messages'][0]['content'],'Current user instructions.')
            self.assertEqual(context.metadata['terminal'],'ready')

    async def test_shell_without_required_approval_still_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.fixture(directory)
            agent.arm = 'persona'; agent.plan_approved = False; agent.approved_command = None
            agent.binding = {'opening_message':'Review the task first.','persona':{'scenario_specific':{
                'terminal':{'command_confirmation':'confirm-each','agent_mode_preference':'plan-first'}}}}
            async def user_response(messages, event, action):
                if action['action']=='ready':agent.terminal='ready';return False
                return True
            agent.user_response = user_response
            env = types.SimpleNamespace(exec=AsyncMock())
            actions = [self.action('shell',command='printf unauthorized'),self.action('ready',message='Review.')]
            with patch.object(tb3,'ReactiveUser',return_value=object()), patch.object(tb3,'call',side_effect=actions):
                await agent.run('unused',env,types.SimpleNamespace(metadata=None))
            env.exec.assert_not_awaited()
            blocked = [args for args in agent.event.call_args_list if args.args[0]=='command_blocked']
            self.assertEqual(len(blocked),1)


if __name__ == '__main__':
    unittest.main()
