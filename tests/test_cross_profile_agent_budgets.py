"""Cross-profile coding-agent output budget wiring and audit coverage."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from swe_persona.worker import install_v6

ROOT = Path(__file__).parents[1]
MINI_SIDECAR = (ROOT/'src/swe_persona/vendor/swe_pro/mini/runtime/pcb_v6/bridge/mini_sidecar.py')


class AdapterBudgetTests(unittest.TestCase):
    def test_text_agent_call_sites_delegate_budget_resolution(self):
        for path in (
            ROOT/'src/swe_persona/vendor/swe_pro/shell/portable/runtime-v3/tb2_persona_pilot/agent.py',
            ROOT/'src/swe_persona/vendor/swe_live/runtime/project/tb2_persona_pilot/agent.py',
        ):
            with self.subTest(path=path):
                source=path.read_text()
                self.assertIn('role=f"agent.turn{turn}", max_output_tokens=None',source)
                self.assertNotIn('role=f"agent.turn{turn}", max_output_tokens=4096',source)

    def test_together_agent_delegates_budget_but_persona_call_keeps_its_fallback(self):
        source=(ROOT/'src/swe_persona/vendor/swe_together/swe_together_pilot/persona_runtime/public_run.py').read_text()
        self.assertIn("api.json,request,AGENT_SCHEMA,'agent_action',None",source)
        self.assertIn("api.json(messages,schema,'persona_decision',16000",source)

    def test_pro_mini_receives_effort_budget_and_explicit_limit_stays_authoritative(self):
        for request_options,expected in (
            ({'reasoning_effort':'max'},131072),
            ({'reasoning_effort':'max','max_tokens':24000},24000),
        ):
            with self.subTest(request_options=request_options), tempfile.TemporaryDirectory() as temp:
                package=ModuleType('pcb_v6');package.__path__=[]
                llm=ModuleType('pcb_v6.llm')
                class Llm:
                    def __init__(self,*args,**kwargs):
                        pass
                llm.Llm=Llm;package.llm=llm
                config={
                    'agent':{'model':'agent','base_url':'https://agent.invalid/v1',
                             'api_key_env':'AGENT_KEY','api':'chat_completions',
                             'max_tokens_field':'max_tokens','request_options':request_options},
                    'judge':{'model':'judge','base_url':'https://judge.invalid/v1',
                             'api_key_env':'JUDGE_KEY','api':'responses','request_options':{}},
                }
                with patch.dict(sys.modules,{'pcb_v6':package,'pcb_v6.llm':llm}), \
                     patch.object(sys,'path',list(sys.path)), \
                     patch.dict(os.environ,{'AGENT_KEY':'secret','JUDGE_KEY':'secret'}):
                    install_v6(Path(temp),config,Path(temp))
                    options=json.loads(os.environ['SWE_PERSONA_AGENT_REQUEST_OPTIONS'])
                    self.assertEqual(options['max_tokens'],expected)
                    self.assertEqual(int(os.environ['SWE_PERSONA_AGENT_OUTPUT_BUDGET']),expected)
                    self.assertEqual(os.environ['SWE_PERSONA_AGENT_REASONING_EFFORT'],'max')


class MiniAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('fixture_mini_sidecar',MINI_SIDECAR)
        cls.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_mini_agent_audit_records_common_fields(self):
        module=self.module
        model=module.RecipientTextModel(
            model_name='openai/fixture',model_kwargs={},
            observation_template='{{output}}',format_error_template='{{error}}',
            cost_tracking='ignore_errors')
        response=NS(model='fixture',
            usage=NS(prompt_tokens=3,completion_tokens=7,
                     completion_tokens_details=NS(reasoning_tokens=2)),
            choices=[NS(finish_reason='stop',
                        message=NS(content='ok',reasoning_content='why'))])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            model.recovery_deadline=time.monotonic()+10
            model.recovery_audit=root/'transport-recovery.jsonl'
            model.output_audit=root/'api-calls.jsonl'
            model.output_budget=131072
            model.output_effort='max'
            with patch.object(module.LitellmTextbasedModel,'_query',return_value=response):
                model._query([])
            audit=json.loads(model.output_audit.read_text())
        self.assertEqual(audit['requested_max_tokens'],131072)
        self.assertEqual(audit['effort'],'max')
        self.assertEqual(audit['finish_reason'],'stop')
        self.assertEqual(audit['reasoning_tokens'],2)
        self.assertEqual(audit['visible_output_chars'],2)


if __name__ == '__main__':
    unittest.main()
