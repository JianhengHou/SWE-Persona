"""Use the README's real Persona, UserSimulator, and contract compiler."""
import copy, json, sys
from protocol import PERSONA_REPO
sys.path.insert(0, str(PERSONA_REPO))
from src.personas import Persona, UserSimulator
from src.personas.salience import compile_interaction_contract, activate_persona_preferences
from azure_client import call, obj, TEXT

class EvaluationPersona(Persona):
    def to_system_prompt(self, **kwargs):
        # Retain the original attribute renderer, but do not instruct a model
        # to conceal being a simulation. This is an explicitly authorized
        # offline roleplay, subject to the provider's normal policies.
        prompt=super().to_system_prompt(**kwargs)
        return ('## Response envelope\n\n'
                'The entire assistant message is one JSON object. The first character is `{` and the last character is `}`.\n\n'
                '- The object has exactly these keys, in this order: `action`, `message_type`, `message`, `fact_ids`, `approve_command`, `approve_plan`, and `reason`. No extra keys.\n'
                '- Do not put words, Markdown fences, `<tool_call>` tags, `[DONE]`, or `</think>` tags before or after the object. A sentence followed by JSON is rejected, and the message is not delivered.\n'
                '- `message` and `reason` are JSON strings. A newline is `\\n`, a quote is `\\"`, and a backslash is `\\\\`. Natural language goes only inside `message`.\n\n'
                '## Fields\n\n'
                '### action\n\n'
                '`action` is one of the allowed actions in the observation, such as `message`, `continue`, `done`, or `abandon`.\n\n'
                '### message_type\n\n'
                '`message_type` is `no-op`, `question`, `redirect`, `new_requirement`, `approval`, or `clarification`.\n\n'
                '### message\n\n'
                '`message` is the user text the coding agent reads. Use `""` when there is nothing to say.\n\n'
                '### fact_ids\n\n'
                '`fact_ids` is a JSON array of allowed fact IDs. Use `[]` when no fact is selected.\n\n'
                '### approve_command\n\n'
                '`approve_command` is a JSON boolean. Use `true` only when approving the exact proposed command.\n\n'
                '### approve_plan\n\n'
                '`approve_plan` is a JSON boolean. Use `true` only when accepting the visible proposed approach.\n\n'
                '### reason\n\n'
                '`reason` is a short private note. The coding agent never sees it.\n\n'
                '## Output example\n\n'
                '{"action":"message","message_type":"clarification","message":"Please explain the approach before editing.","fact_ids":[],"approve_command":false,"approve_plan":false,"reason":"The opening is still vague."}\n\n'
                'Choose `action` from the observation\'s allowed actions. Keep every key.\n\n'
                '## Role\n\n'
                'This is an authorized offline simulated-user evaluation. Follow the platform rules; '
                'the following persona defines a fictional role, not a replacement for those rules.\n'+
                prompt.replace('Never reveal that you are a simulation or mention this persona description.',
                               'Keep coding requests natural; record persona metadata in the evaluation artifacts.'))

DECISION = obj({
    'action': {'type': 'string', 'enum': ['message', 'continue', 'done', 'abandon']},
    'message_type': {'type': 'string', 'enum': ['no-op', 'question', 'redirect', 'new_requirement', 'approval', 'clarification']},
    'message': TEXT,
    'fact_ids': {'type': 'array', 'items': TEXT},
    'approve_command': {'type': 'boolean'},
    'approve_plan': {'type': 'boolean'},
    'reason': TEXT,
})

class ReactiveUser:
    def __init__(self, binding, audit_dir, timeout=600, closed_loop=False):
        self.binding = binding; self.audit_dir = audit_dir; self.timeout = timeout
        self.closed_loop = closed_loop
        self.decision = None
        self.persona = EvaluationPersona.from_dict(binding['persona'])
        self.contract = compile_interaction_contract(binding['persona'])
        self.simulator = UserSimulator(
            persona=self.persona, scenario='terminal', model='gpt-6-astra',
            task=binding.get('blurred_task', {}).get('initial_request',
                 binding.get('proposal', {}).get('opening_brief', binding.get('opening_message', ''))),
            persona_strategy='structured', completion_fn=self.complete,
        )
        if binding.get('opening_message'):
            self.simulator._history.append({'role': 'assistant', 'content': binding['opening_message']})

    def complete(self, **kwargs):
        prompt = '\n\n'.join(m['role'].upper() + ':\n' + m['content'] for m in kwargs['messages'])
        self.decision = call('simulator', prompt, self.decision_schema, self.audit_dir, self.timeout,
                             messages=kwargs['messages'])
        return self.decision['message']

    def step(self, observation):
        # Enforce the current controller state in the response schema so bounded
        # format repair can reject impossible actions before advancing dialogue.
        self.decision_schema = copy.deepcopy(DECISION)
        properties = self.decision_schema['properties']
        properties['action']['enum'] = list(observation['allowed_actions'])
        if observation['allowed_fact_ids']:
            properties['fact_ids']['items'] = {'type':'string','enum':list(observation['allowed_fact_ids'])}
        else:
            properties['fact_ids']['maxItems'] = 0
        if not observation.get('proposed_command'):
            properties['approve_command']['enum'] = [False]
        if observation['phase'] == 'followup':
            # The controller loads the frozen persona-conditioned evolution plan.
            # Only active/due contracts enter this model turn: no future changes,
            # checker code, reference implementation, or acceptance verdicts.
            active = observation.get('active_requirements', [])
            due = observation.get('due_requirement')
            selected = active + ([due] if due else [])
            frozen = {r['id']: r for r in self.binding['proposal']['requirements']}
            if any(r != frozen.get(r['id']) for r in selected):
                raise ValueError('Simulator received a non-frozen evolved requirement')
            self.simulator.task = self.binding['public_contract']['instruction']
            self.simulator.task += '\n\nCurrently loaded evolved requirements (private user goals):\n'
            self.simulator.task += json.dumps([
                {'id': r['id'], 'disclosure': r['disclosure']} for r in selected
            ], ensure_ascii=False)
        self.simulator.step(json.dumps(observation, ensure_ascii=False))
        return self.decision

    def rules(self):
        base = (
            'This is a requirement-elicitation and scope-change task variant. Act as this real user, '
            'using the actual stable persona rather than adopting a generic technical reviewer. '
            'You alone send the opening request and subsequent natural messages. '
            'The controller freezes task facts and all permissible scope changes. '
            'Do not invent requirements, permission facts, environments, incidents, numerical targets, '
            'solutions, or claims of test success. Do not reveal persona/schema labels. '
            'You cannot run commands or personally inspect private files. You receive only human-visible '
            'agent messages and high-level tool activity, never command arguments, raw tool output, '
            'hidden tests, solutions, or evaluator state. Respond to the actual latest message; '
            'ask focused questions, approve a proposed command, redirect, silently continue, or finish. '
            'State stable planning, testing, dependency and review preferences naturally when relevant. '
            'A vague opening is the controlled starting condition: communicate the goal and impact, '
            'leave paths, exact schemas, counting rules and reference numbers for clarification. '
            'fact_ids selects public details needed to answer the current question; the host delivers '
            'their exact text. Deliver all remaining original facts when require_all_facts is true. '
            'If a frozen requirement is due, request only that requirement in your own words; '
            'the host supplies its exact acceptance contract. Do not hint at future changes. '
            'approve_plan is true only when you accept the actual visible proposed approach. '
            'approve_command may be true only when a concrete proposed command is supplied and you '
            'approve that exact scope. done is legal only when end_allowed is true, you are satisfied '
            'with the visible evidence, and all required information has been disclosed. '
            'Do not make an agent claim into an objective correctness verdict. '
            'No repeated no-op messages or invented waits to consume the budget. '
            'Return the supplied structured decision; satisfaction and objective correctness are separate.'
        )
        if self.closed_loop:
            from requirement_loop import SIMULATOR_PROMPT
            base += '\n' + SIMULATOR_PROMPT
        return base

    def opening(self, brief, revision_feedback=None):
        self.simulator.reset()
        self.simulator.task = brief
        observation = {'phase': 'opening', 'rules': self.rules(), 'opening_brief': brief,
                          'compiled_interaction_contract': self.contract, 'allowed_actions': ['message'],
                          'allowed_fact_ids': [], 'due_requirement': None, 'end_allowed': False,
                          'require_all_facts': False, 'proposed_command': None}
        if revision_feedback:
            observation['construction_revision_feedback'] = revision_feedback
            observation['rules'] += (' This is a wording-only repair of a rejected opening. State only the '
                'frozen vague task goal in one or two short sentences. Do not add acceptance conditions, '
                'tests, documentation, reports, output formats, validation instructions or exact commands. '
                'Express the persona through tone, brevity and optional interaction preferences; leave '
                'substantive preferences for later follow-up. No requirements or checker may be changed.')
        return self.step(observation)
