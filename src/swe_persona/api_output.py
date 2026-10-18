"""Bounded completion recovery; configured limits are hard ceilings."""
import re
from dataclasses import dataclass

POLICY = 'bounded-output-recovery-v1'
TOKEN_FIELDS = ('max_tokens', 'max_completion_tokens', 'max_output_tokens')
LENGTH_REASONS = frozenset(('length', 'max_tokens', 'max_output_tokens', 'max_completion_tokens'))
STEPS = (8192, 16384, 32768, 65536, 131072)
EFFORT_BUDGETS = {'none': 8192, 'disabled': 8192, 'minimal': 8192,
                 'low': 16384, 'medium': 65536, 'high': 65536,
                 'xhigh': 131072, 'max': 131072}


class IncompleteOutputError(ValueError):
    """An explicitly truncated generation must not be used as a JSON action."""


def configured_limit(options):
    limits = [options[k] for k in TOKEN_FIELDS if k in options]
    if len(limits) > 1:
        raise ValueError('Specify only one output token limit')
    if not limits:
        return None
    value = limits[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('Output token limit must be a positive integer')
    return value


def configured_effort(options):
    """Return one shared reasoning-effort value from provider request options."""
    reasoning = options.get('reasoning') or {}
    return options.get('reasoning_effort', reasoning.get('effort'))


def default_budget(effort):
    return EFFORT_BUDGETS.get(str(effort).lower() if effort is not None else '',
                             EFFORT_BUDGETS['medium'])


def is_truncated(reason):
    return str(reason or '').lower() in LENGTH_REASONS


def provider_ceiling(status, message, current):
    """Recognize token-range errors only; schema/auth errors never qualify.

    Prefer a bound attached to the token parameter. An ambiguous range error
    permits one smaller request, never an increase or a format downgrade.
    """
    text = str(message or '').lower()
    if status != 400 or not any(k in text for k in TOKEN_FIELDS):
        return None
    if not any(p in text for p in ('range', 'maximum', 'at most', 'less than',
                                  'cannot exceed', 'must be <=', 'must be between',
                                  'too large', 'too high', 'should be <=')):
        return None
    # Restrict parsing to the clause naming the token parameter.
    start = min(text.find(k) for k in TOKEN_FIELDS if k in text)
    clause = text[start:]
    patterns = (r'\[\s*\d+\s*,\s*(\d+)\s*\]',
                r'between\s+\d+\s+and\s+(\d+)',
                r'(?:maximum(?:\s+(?:value|allowed))?(?:\s+is)?|at most|<=|less than or equal to|cannot exceed)\s*[:=]?\s*(\d+)')
    for pattern in patterns:
        match = re.search(pattern, clause)
        if match:
            bound = int(match.group(1))
            return bound if 0 < bound < current else None
    return max((step for step in STEPS if step < current), default=max(1, current // 2)) if current > 1 else None


@dataclass
class OutputBudget:
    current: int
    explicit: bool = False
    ceiling: int | None = None
    raised: bool = False
    shrinks: int = 0

    @classmethod
    def resolve(cls, *, adaptive, effort=None, configured=None, requested=None, ceiling=None):
        explicit = configured is not None or requested is not None
        current = configured if configured is not None else requested
        if current is None:
            current = default_budget(effort) if adaptive else 4096
        if isinstance(current, bool) or not isinstance(current, int) or current <= 0:
            raise ValueError('Output token limit must be a positive integer')
        return cls(min(current, ceiling) if ceiling else current,
                   explicit=explicit or not adaptive, ceiling=ceiling)

    def grow(self, reason):
        if self.explicit or self.raised or not is_truncated(reason):
            return False
        next_step = next((step for step in STEPS if step > self.current), None)
        if next_step is None or (self.ceiling is not None and next_step > self.ceiling):
            return False
        self.current = next_step
        self.raised = True
        return True

    def shrink(self, status, message):
        if self.shrinks >= 4:
            return False
        ceiling = provider_ceiling(status, message, self.current)
        if ceiling is None:
            return False
        self.ceiling = ceiling if self.ceiling is None else min(self.ceiling, ceiling)
        self.current = min(self.current, self.ceiling)
        self.shrinks += 1
        return True


def coding_budget(options, *, effort=None, requested=None, ceiling=None):
    """Resolve the common coding-agent budget independently of response format."""
    return OutputBudget.resolve(adaptive=True,
        effort=configured_effort(options) if effort is None else effort,
        configured=configured_limit(options), requested=requested, ceiling=ceiling)
