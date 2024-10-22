from typing import Dict, Type

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChitChatAnswerCommand,
    Command,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    SessionStartCommand,
    SkipQuestionCommand,
    RestartCommand,
)
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.chitchat import ChitchatPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.restart import RestartPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.session_start import (
    SessionStartPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.skip_question import (
    SkipQuestionPatternFlowStackFrame,
)

triggerable_pattern_to_command_class: Dict[str, Type[Command]] = {
    SessionStartPatternFlowStackFrame.flow_id: SessionStartCommand,
    CancelPatternFlowStackFrame.flow_id: CancelFlowCommand,
    ChitchatPatternFlowStackFrame.flow_id: ChitChatAnswerCommand,
    HumanHandoffPatternFlowStackFrame.flow_id: HumanHandoffCommand,
    SearchPatternFlowStackFrame.flow_id: KnowledgeAnswerCommand,
    SkipQuestionPatternFlowStackFrame.flow_id: SkipQuestionCommand,
    CannotHandlePatternFlowStackFrame.flow_id: CannotHandleCommand,
    RestartPatternFlowStackFrame.flow_id: RestartCommand,
}
