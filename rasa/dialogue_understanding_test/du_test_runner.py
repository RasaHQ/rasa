import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import structlog
from tqdm import tqdm

from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.core.exceptions import AgentNotReady
from rasa.core.persistor import StorageType
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)
from rasa.dialogue_understanding_test.test_case_simulation.test_case_tracker_simulator import (  # noqa: E501
    TestCaseTrackerSimulator,
    TestCaseTrackerSimulatorException,
    TestCaseTrackerSimulatorResult,
)
from rasa.dialogue_understanding_test.utils import filter_metadata
from rasa.e2e_test.e2e_test_case import (
    KEY_STUB_CUSTOM_ACTIONS,
    Fixture,
    Metadata,
)
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.shared.agents.agent_setup import AgentsConnectionCleanup
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import PREDICTED_COMMANDS, PROMPTS
from rasa.shared.utils.llm import create_tracker_for_user_step
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.agent import Agent

structlogger = structlog.get_logger()


class DialogueUnderstandingTestRunner:
    """Dialogue Understanding test suite runner.

    Responsible for executing dialogue understanding test cases by simulating
    conversations and validating expected outputs against actual responses.

    Attributes:
        agent: The Rasa agent instance used for processing messages
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[StorageType] = None,
        endpoints: Optional[AvailableEndpoints] = None,
        sub_agents_path: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initializes the Dialogue Understanding test suite runner.

        Args:
            model_path: Path to the model.
            model_server: Model server configuration.
            remote_storage: Remote storage to use for model retrieval.
            endpoints: Endpoints configuration.
        """
        import rasa.core.agent

        self._check_action_server(endpoints)
        sub_agents = Configuration.get_instance().available_agents

        async def load_agent_with_agents_connection_cleanup() -> "Agent":
            # Create a wrapper coroutine that handles graceful agents connection
            # cleanup, this prevents abrupt closure when asyncio.run finishes.
            async with AgentsConnectionCleanup():
                return await rasa.core.agent.load_agent(
                    model_path=model_path,
                    model_server=model_server,
                    remote_storage=remote_storage,
                    endpoints=endpoints,
                    sub_agents=sub_agents,
                )
            # Defensive return for mypy - never actually reached
            return  # type: ignore[return-value]

        self.agent = asyncio.run(load_agent_with_agents_connection_cleanup())
        if not self.agent.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. "
                "Please check that the agent was able to "
                "load the trained model."
            )

    @staticmethod
    def _check_action_server(endpoints: AvailableEndpoints) -> None:
        """Check if the action server is reachable."""
        are_custom_actions_stubbed = (
            endpoints
            and endpoints.action
            and endpoints.action.kwargs.get(KEY_STUB_CUSTOM_ACTIONS)
        )
        if endpoints and not are_custom_actions_stubbed:
            E2ETestRunner._action_server_is_reachable(
                endpoints, "dialogue_understanding_test_runner"
            )

    async def run_test_cases(
        self,
        test_cases: List[DialogueUnderstandingTestCase],
        fixtures: List[Fixture],
        metadata: List[Metadata],
    ) -> List[DialogueUnderstandingTestResult]:
        """Run the dialogue understanding tests.

        Args:
            test_cases: List of test cases.
            fixtures: List of fixtures.
            metadata: List of metadata.

        Returns:
            List[DialogueUnderstandingTestResult]: List of test results.
        """
        results = []
        output_channel = CollectingOutputChannel()

        structlogger.info("Starting dialogue understanding tests.")

        for i in tqdm(range(len(test_cases))):
            test_case = test_cases[i]

            # set up the tracker by simulating the conversation
            tracker_simulator = TestCaseTrackerSimulator(self.agent, test_case)
            await tracker_simulator.initialize_tracker(fixtures)
            try:
                simulation_result = await tracker_simulator.simulate_test_case(metadata)
            except TestCaseTrackerSimulatorException as e:
                structlogger.error(
                    "Failed to simulate test case. Skipping test case.",
                    error=str(e),
                )
                continue

            # run the actual test case
            test_result = await self.run_test_case(
                test_case, metadata, simulation_result, output_channel
            )
            results.append(test_result)

        structlogger.info("Finished dialogue understanding tests.")

        return results

    async def run_test_case(
        self,
        test_case: DialogueUnderstandingTestCase,
        metadata: List[Metadata],
        simulation_result: TestCaseTrackerSimulatorResult,
        output_channel: CollectingOutputChannel,
    ) -> DialogueUnderstandingTestResult:
        """Runs dialogue understanding test case.

        Args:
            test_case: The test case to run.
            metadata: List of metadata.
            simulation_result: The test case tracker.
            output_channel: The output channel.

        Returns:
            A dialogue understanding test result.
        """
        test_passed = True

        sender_id = simulation_result.sender_id
        user_uttered_event_indices = simulation_result.user_uttered_event_indices
        test_case_tracker = await self.agent.tracker_store.retrieve(sender_id)

        for user_step_index, user_step in enumerate(
            test_case.iterate_over_user_steps()
        ):
            if user_step_index >= len(user_uttered_event_indices):
                # the conversation could not be simulated completely and was
                # stopped early, no further user messages can be evaluated
                return DialogueUnderstandingTestResult(
                    test_case=test_case, passed=test_passed
                )

            # create and save the tracker at the time just
            # before the user message was sent
            step_sender_id = f"{sender_id}_{user_step_index}"
            await create_tracker_for_user_step(
                step_sender_id,
                self.agent,
                test_case_tracker,
                user_uttered_event_indices[user_step_index],
            )

            # Total latency of a message roundtrip
            latency = None

            # send the user message
            try:
                start = time.time()
                await self._send_user_message(
                    step_sender_id,
                    test_case,
                    user_step,
                    metadata,
                    output_channel=output_channel,
                )
                end = time.time()
                latency = end - start
            except Exception as e:
                structlogger.error(
                    "dialogue_understanding_test_runner.send_user_message.failed",
                    test_case=test_case.full_name(),
                    user_message=user_step.text,
                    error=str(e),
                )
                # as sending the user message failed, we cannot continue with the test
                # as subsequent user message do not match the actual conversation
                # return a test result up until this point
                return DialogueUnderstandingTestResult(
                    test_case=test_case, passed=test_passed
                )

            # get the dialogue understanding output
            tracker = await self.agent.tracker_store.retrieve(step_sender_id)
            dialogue_understanding_output = self.get_dialogue_understanding_output(
                tracker, user_uttered_event_indices[user_step_index], latency
            )
            user_step.dialogue_understanding_output = dialogue_understanding_output

            # check if we have a command match
            if not user_step.has_passed():
                test_passed = False

        return DialogueUnderstandingTestResult(test_case=test_case, passed=test_passed)

    def get_dialogue_understanding_output(
        self,
        tracker: DialogueStateTracker,
        index_user_uttered_event: int,
        latency: Optional[float] = None,
    ) -> Optional[DialogueUnderstandingOutput]:
        """Returns the dialogue understanding output.

        Creates the dialogue understanding output from the commands and prompts
        added to the user uttered event.

        Args:
            tracker: The tracker for the current test step.
            index_user_uttered_event: The index of the user uttered event.

        Returns:
        The dialogue understanding output with commands and optionally prompts.
        """
        user_uttered_event = self._get_user_uttered_event_from_tracker(
            tracker, index_user_uttered_event
        )
        if user_uttered_event is None:
            return None

        predicted_commands: Dict[str, Any] = user_uttered_event.parse_data.get(  # type:ignore[assignment]
            PREDICTED_COMMANDS, {}
        )
        # convert the predicted commands to Command objects
        commands = {}
        for component, list_of_commands in predicted_commands.items():
            # remove any duplicate commands
            commands[component] = list(
                set(
                    [Command.command_from_json(command) for command in list_of_commands]
                )
            )

        return DialogueUnderstandingOutput(
            commands=commands,
            prompts=user_uttered_event.parse_data.get(PROMPTS, []),
            latency=latency,
        )

    @staticmethod
    def _get_user_uttered_event_from_tracker(
        tracker: DialogueStateTracker,
        index_user_uttered_event: int,
    ) -> Optional[UserUttered]:
        """Returns the user uttered event from the tracker at the given index."""
        if (
            tracker.events is None
            or len(list(tracker.events)) <= index_user_uttered_event
        ):
            return None

        user_uttered_event = list(tracker.events)[index_user_uttered_event]

        if not user_uttered_event or not isinstance(user_uttered_event, UserUttered):
            return None

        return user_uttered_event

    async def _send_user_message(
        self,
        sender_id: str,
        test_case: DialogueUnderstandingTestCase,
        user_step: DialogueUnderstandingTestStep,
        metadata: List[Metadata],
        output_channel: CollectingOutputChannel = CollectingOutputChannel(),
    ) -> None:
        """Sends a user message to the agent."""
        metadata_for_step = filter_metadata(test_case, user_step, metadata, sender_id)

        user_message = UserMessage(
            user_step.text,
            output_channel,
            sender_id,
            metadata=metadata_for_step,
        )

        with set_record_commands_and_prompts():
            await self.agent.handle_message(user_message)
