from typing import Dict, List, Set, Type

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChitChatAnswerCommand,
    Command,
    CorrectSlotsCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    RestartCommand,
    SessionStartCommand,
    SetSlotCommand,
    SkipQuestionCommand,
)
from rasa.dialogue_understanding.commands.user_silence_command import UserSilenceCommand
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
from rasa.dialogue_understanding.patterns.user_silence import (
    UserSilencePatternFlowStackFrame,
)

triggerable_pattern_to_command_class: Dict[str, Type[Command]] = {
    SessionStartPatternFlowStackFrame.flow_id: SessionStartCommand,
    UserSilencePatternFlowStackFrame.flow_id: UserSilenceCommand,
    CancelPatternFlowStackFrame.flow_id: CancelFlowCommand,
    ChitchatPatternFlowStackFrame.flow_id: ChitChatAnswerCommand,
    HumanHandoffPatternFlowStackFrame.flow_id: HumanHandoffCommand,
    SearchPatternFlowStackFrame.flow_id: KnowledgeAnswerCommand,
    SkipQuestionPatternFlowStackFrame.flow_id: SkipQuestionCommand,
    CannotHandlePatternFlowStackFrame.flow_id: CannotHandleCommand,
    RestartPatternFlowStackFrame.flow_id: RestartCommand,
}


def filter_slot_commands(
    commands: List[Command], overlapping_slot_names: Set[str]
) -> List[Command]:
    """Filter out slot commands that set overlapping slots."""
    filtered_commands = []

    for command in commands:
        if (
            isinstance(command, SetSlotCommand)
            and command.name in overlapping_slot_names
        ):
            continue

        if isinstance(command, CorrectSlotsCommand):
            allowed_slots = [
                slot
                for slot in command.corrected_slots
                if slot.name not in overlapping_slot_names
            ]
            if not allowed_slots:
                continue

            command.corrected_slots = allowed_slots

        filtered_commands.append(command)

    return filtered_commands
