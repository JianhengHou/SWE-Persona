"""A checker driver's CHECK line must survive evidence text that quotes UTF-8-decoded binary files."""
import ast
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional
import unittest

ROOT = Path(__file__).parents[1]
VENDOR = ROOT / 'src/swe_persona/vendor'
SHELL = VENDOR / 'swe_pro/shell/portable/runtime-v3'
PARSERS = (SHELL / 'runner/checkers.py', SHELL / 'tb2_persona_pilot/checkers.py',
           VENDOR / 'swe_live/runtime/project/tb2_persona_pilot/checkers.py')
DRIVER = SHELL / 'runner/fixtures/checker_driver.py'


def function(path, name):
    """Compile one real top-level function without importing its runtime."""
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'json': json, 'Any': Any, 'Dict': Dict, 'Optional': Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def emitted(**evidence):
    out = io.StringIO()
    with redirect_stdout(out):
        function(DRIVER, 'emit')('pass', 'evidence_collected', **evidence)
    return out.getvalue()


class CheckLineTests(unittest.TestCase):
    # Redis dump/AOF files left in /app, read with errors="replace", can hold these code points.
    TEXTS = ('plain text', b'dump \xc2\x85 bytes'.decode('utf-8', 'replace'),
             b'aof \xe2\x80\xa8 bytes'.decode('utf-8', 'replace'), b'aof \xe2\x80\xa9 bytes'.decode('utf-8', 'replace'))

    def test_verdict_is_parsed_whatever_the_evidence_text(self):
        for path in PARSERS:
            parse = function(path, '_parse_check_line')
            for text in self.TEXTS:
                with self.subTest(parser=path.relative_to(VENDOR), text=text):
                    stdout = 'driver log\n' + emitted(files=[{'path': '/app/dump.rdb', 'text': text}])
                    payload = parse(stdout)
                    self.assertIsNotNone(payload)
                    self.assertEqual(payload['status'], 'pass')
                    self.assertEqual(payload['evidence']['files'][0]['text'], text)

    def test_the_last_check_line_still_wins(self):
        for path in PARSERS:
            with self.subTest(parser=path.relative_to(VENDOR)):
                first = emitted(files=[{'path': 'a', 'text': 'first'}])
                last = emitted(files=[{'path': 'b', 'text': self.TEXTS[2]}])
                self.assertEqual(function(path, '_parse_check_line')(first + last)['evidence']['files'][0]['path'], 'b')


if __name__ == '__main__':
    unittest.main()
