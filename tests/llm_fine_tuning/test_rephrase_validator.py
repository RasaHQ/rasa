from typing import List, Text
from unittest.mock import MagicMock, Mock, patch

import pytest

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    KnowledgeAnswerCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.paraphrasing.rephrase_validator import RephraseValidator
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


@pytest.fixture
def validator() -> RephraseValidator:
    flows = flows_from_str(
        """
        flows:
          transfer_money:
            description: send money to a recipient
            steps:
            - collect: recipient
            - collect: amount
          book_hotel:
            description: book a hotel
            steps:
            - collect: hotel_name
            - collect: start_date
            - collect: end_date
        """
    )
    return RephraseValidator(flows)


@pytest.fixture
def sender_id() -> Text:
    return "unit_test_rephrase_validator"


@pytest.fixture
def tracker(sender_id: Text) -> DialogueStateTracker:
    events = [
        UserUttered("I want to send money to John"),
        BotUttered("How much money do you want to send?"),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id=sender_id, evts=events, slots=[]
    )
    return tracker


@pytest.fixture
def conversation(tracker: DialogueStateTracker) -> Conversation:
    test_case = TestCase.from_dict(
        {
            "test_case": "transfer_money",
            "steps": [
                {"user": "I want to send money to John"},
                {"bot": "How much money do you want to send?"},
            ],
        }
    )

    return Conversation(
        test_case.name,
        test_case,
        [
            ConversationStep(
                test_case.steps[0],
                [StartFlowCommand("transfer_money")],
                """
                Here is what happened previously in the conversation:
                USER: I want to send money to John
                AI: How much money do you want to send?
                ===
                The user just said '''I want to send money to John'''.
                """,
                tracker_event_index=0,
            ),
            test_case.steps[1],
        ],
        "transcript",
        tracker,
    )


@pytest.fixture
def rephrased_user_messages() -> List[RephrasedUserMessage]:
    return [
        RephrasedUserMessage(
            "I want to send money to John",
            ["Send money to John", "Transfer money to John"],
        )
    ]


@pytest.mark.asyncio
async def test_validate_rephrasings_passing(
    compact_agent: Agent,
    validator: RephraseValidator,
    conversation: Conversation,
    rephrased_user_messages: List[RephrasedUserMessage],
):
    with patch.object(validator, "_validate_rephrase_is_passing", return_value=True):
        validated_rephrasings = await validator.validate_rephrasings(
            compact_agent, rephrased_user_messages, conversation
        )

        assert len(validated_rephrasings[0].passed_rephrasings) == 2
        assert "Send money to John" in validated_rephrasings[0].passed_rephrasings
        assert "Transfer money to John" in validated_rephrasings[0].passed_rephrasings
        assert len(validated_rephrasings[0].failed_rephrasings) == 0


@pytest.mark.asyncio
async def test_validate_rephrasings_failing(
    compact_agent: Agent,
    validator: RephraseValidator,
    conversation: Conversation,
    rephrased_user_messages: List[RephrasedUserMessage],
):
    with patch.object(validator, "_validate_rephrase_is_passing", return_value=False):
        validated_rephrasings = await validator.validate_rephrasings(
            compact_agent, rephrased_user_messages, conversation
        )

        assert len(validated_rephrasings[0].failed_rephrasings) == 2
        assert "Send money to John" in validated_rephrasings[0].failed_rephrasings
        assert "Transfer money to John" in validated_rephrasings[0].failed_rephrasings
        assert len(validated_rephrasings[0].passed_rephrasings) == 0


@patch(
    "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator.SingleStepLLMCommandGenerator."
    "invoke_llm"
)
@pytest.mark.asyncio
async def test_rephrase_is_passing(
    mock_invoke_llm: Mock,
    single_step_agent: Agent,
    validator: RephraseValidator,
    conversation: Conversation,
):
    mock_invoke_llm.return_value = "StartFlow(transfer_money)"

    rephrase = "I want to transfer some money to John"
    passing = await validator._validate_rephrase_is_passing(
        single_step_agent,
        rephrase,
        conversation.steps[0],
        conversation.name,
        conversation.tracker,
    )

    assert passing is True


@pytest.mark.asyncio
@patch(
    "rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator.CompactLLMCommandGenerator.invoke_llm"
)
async def test_rephrase_is_passing_using_compact_llm_command_generator(
    mock_invoke_llm: Mock,
    compact_agent: Agent,
    validator: RephraseValidator,
    conversation: Conversation,
):
    # Set syntax version to v2. This is required for the CompactLLMCommandGenerator
    # to work. When the agent loads, the CompactLLMCommandGenerator is initialized
    # and the command syntax is set to v2 automatically. However, in the tests, it is
    # not loaded, so we need to set it manually.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)
    mock_invoke_llm.return_value = "start flow transfer_money"

    rephrase = "I want to transfer some money to John"
    passing = await validator._validate_rephrase_is_passing(
        compact_agent,
        rephrase,
        conversation.steps[0],
        conversation.name,
        conversation.tracker,
    )

    assert passing is True

    # Reset the syntax version. This is required to avoid side effects in other tests.
    CommandSyntaxManager.reset_syntax_version()


@patch(
    "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator.SingleStepLLMCommandGenerator.invoke_llm"
)
@pytest.mark.asyncio
async def test_rephrase_is_not_passing(
    mock_invoke_llm: Mock,
    single_step_agent: Agent,
    validator: RephraseValidator,
    conversation: Conversation,
):
    mock_invoke_llm.return_value = "SetSlot('recipient', 'John')"

    rephrase = "I want to transfer some money to John"
    passing = await validator._validate_rephrase_is_passing(
        single_step_agent,
        rephrase,
        conversation.steps[0],
        conversation.name,
        conversation.tracker,
    )

    assert passing is False


@pytest.mark.parametrize(
    "expected_commands, actual_commands, match",
    [
        ([StartFlowCommand("foo")], [StartFlowCommand("foo")], True),
        (
            [ClarifyCommand(options=["a", "b", "c"])],
            [ClarifyCommand(options=["a", "b", "c"])],
            True,
        ),
        ([CancelFlowCommand()], [CancelFlowCommand()], True),
        ([SkipQuestionCommand()], [SkipQuestionCommand()], True),
        ([ChitChatAnswerCommand()], [ChitChatAnswerCommand()], True),
        ([KnowledgeAnswerCommand()], [KnowledgeAnswerCommand()], True),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("foo", "bar")], True),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("bar", "foo")], False),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("foo", "BAR")], True),
        (
            [ChitChatAnswerCommand(), StartFlowCommand("foo")],
            [ChitChatAnswerCommand()],
            False,
        ),
        (
            [KnowledgeAnswerCommand()],
            [KnowledgeAnswerCommand(), StartFlowCommand("foo")],
            False,
        ),
    ],
)
def test_commands_match(
    validator: RephraseValidator,
    expected_commands: List[Command],
    actual_commands: List[Command],
    match: bool,
):
    assert validator._check_commands_match(expected_commands, actual_commands) is match


@pytest.mark.asyncio
async def test_send_rephrased_message_to_agent():
    rephrase = "I'd like to send money to John."
    step = MagicMock()
    step.tracker_event_index = None

    previous_tracker = DialogueStateTracker("old_sender_id", slots=[])

    # minimal agent with in-memory tracker store
    domain = Domain.empty()
    agent = Agent(domain=domain)
    agent.tracker_store = InMemoryTrackerStore(domain)

    async def _fake_handle(msg: UserMessage):
        t = DialogueStateTracker(msg.sender_id, slots=[])
        t.update(UserUttered(msg.text))
        await agent.tracker_store.save(t)

    agent.handle_message = _fake_handle
    test_case_name = "test_send_rephrased_message_to_agent"

    returned_tracker = await RephraseValidator._send_rephrased_message_to_agent(
        rephrase,
        step,
        test_case_name=test_case_name,
        agent=agent,
        tracker=previous_tracker,
    )

    assert test_case_name in returned_tracker.sender_id
    assert returned_tracker.latest_message.text == rephrase
