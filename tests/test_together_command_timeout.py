"""Harbor command timeouts must stay inside the episode."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

ROOT = Path(__file__).parents[1]
RUNTIME = ROOT/'src/swe_persona/vendor/swe_together/swe_together_pilot/persona_runtime'
package = ModuleType('fixture_together_command_timeout')
package.__path__ = [str(RUNTIME)]
sys.modules[package.__name__] = package
public = importlib.import_module(package.__name__+'.public_run')


class TogetherCommandTimeoutTests(unittest.TestCase):
    def test_harbor_runtime_error_counts_as_a_command_timeout(self):
        self.assertTrue(public.command_timed_out(TimeoutError()))
        self.assertTrue(public.command_timed_out(RuntimeError('Command timed out after 180 seconds')))

    def test_other_runtime_errors_still_escape(self):
        self.assertFalse(public.command_timed_out(RuntimeError('container exited')))
        self.assertFalse(public.command_timed_out(RuntimeError('Command timed out')))
