import asyncio
from typing import Dict, Optional, Text, Union, List

import structlog

from rasa.core.exceptions import AgentNotReady
from rasa.core.persistor import StorageType
from rasa.core.utils import AvailableEndpoints
from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestCase
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)
from rasa.e2e_test.e2e_test_case import (
    KEY_STUB_CUSTOM_ACTIONS,
    ActualStepOutput,
    TestStep,
    Fixture,
    Metadata,
)
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger()

TEST_TURNS_TYPE = Dict[int, Union[TestStep, ActualStepOutput]]


class DialogueUnderstandingTestRunner:
    """Dialogue Understanding test suite runner."""

    def __init__(
        self,
        model_path: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[StorageType] = None,
        endpoints: Optional[AvailableEndpoints] = None,
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

        self.agent = asyncio.run(
            rasa.core.agent.load_agent(
                model_path=model_path,
                model_server=model_server,
                remote_storage=remote_storage,
                endpoints=endpoints,
            )
        )
        if not self.agent.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. "
                "Please check that the agent was able to "
                "load the trained model."
            )

    def _check_action_server(self, endpoints: AvailableEndpoints) -> None:
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

    async def run_tests(
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
        return []
