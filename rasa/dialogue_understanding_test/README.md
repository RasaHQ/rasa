# Dialogue Understanding Tests

Dialogue Understanding Tests (DUT) are designed to evaluate the command prediction accuracy of a
chatbot's [dialogue understanding](https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding) module.
Rather than merely assessing whether a chatbot behaves as expected in
[end-to-end (E2E) tests](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#end-to-end-testing), these
tests delve deeper into understanding *why* a chatbot may not be performing as anticipated.
They aim to identify discrepancies between the expected and predicted
[commands](https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding#command-reference) during a conversation,
providing insights into potential pitfalls in the
[command generator](https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding#commandgenerator)'s operation.

The primary focus of Dialogue Understanding Tests is the command generator,
a core component responsible for interpreting user input and orchestrating the chatbot's subsequent actions.
When updates are made to the command generator — such as switching to a different LLM or tweaking the prompt —
Dialogue Understanding Tests offer a structured approach to evaluate how accurately these changes affect command
predictions.

*Note*: Dialogue Understanding Tests only work for [CALM-based assistants](https://rasa.com/docs/rasa-pro/calm)!

## How Dialogue Understanding Tests work

Dialogue Understanding Tests are designed to evaluate a chatbot's dialogue understanding capabilities
within a given conversational context.
In order to run Dialogue Understanding Tests, you first need to write test cases.
Each test case consists of a sequence of interactions that simulate conversations with the chatbot.
These interactions are broken down into user inputs, expected commands, and bot responses.
Each step of a test case is evaluated independently.
The predicted commands of the dialogue understanding module are compared with the expected commands
defined in the test case.
The Dialogue Understanding Test framework is able to evaluate each step of a test case, also if
a previous test step failed.
This allows for a more detailed analysis of the chatbot's performance.
After all test cases have been executed, a detailed report is generated, including metrics such as accuracy,
precision, recall, and f1-score for all commands.

## Running Dialogue Understanding Tests

Dialogue Understanding Tests are hidden behind a feature flag. To enable the feature, set the
environment variable `RASA_PRO_BETA_DIALOGUE_UNDERSTANDING_TEST` to `true`.

To run Dialogue Understanding Tests, execute the following command:

```bash
rasa test du <path-to-test-cases>
```

By default, the test cases are expected to be located in the `dialogue_understanding_tests` directory.
Execute `rasa test du --help` to see the available options for running Dialogue Understanding Tests.

In order to execute any custom action that is needed by the Dialogue Understanding Tests, you need to
either start the action server in the background before running the tests via `rasa run actions` or use
[Stubbing Custom Actions](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#stubbing-custom-actions).

## Defining a Dialogue Understanding Test Case

A test case is structured as a sequence of interactions between a user and the chatbot, specifying both inputs and
expected outcomes. Each test case is composed of multiple steps, and each step consists of:

- **User Utterance**: The input message from the user.
- **Commands**: All expected commands that are generated in response to the user's message.
- **Bot Response(s)**: All expected responses from the bot, which can be either a direct textual response or a reference
  to
  a predefined bot response template.

Here is a sample test case:

```yaml
test_cases:
  - test_case: user_adds_contact_to_their_list
    steps:
      - user: I want to add someone to my contact list
        commands:
          - StartFlow(add_contact)
      - utter: utter_ask_add_contact_handle
      - user: it's @barts
        commands:
          - SetSlot(handle, @barts)
      - bot: "What is the name of the contact you want to add?"
      - user: just Bart
        commands:
          - SetSlot(name, Bart)
```

*Note*: The list of commands and the list of bot responses need to be complete!

[Fixtures](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#fixtures-for-pre-filled-slots),
[Metadata](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#metadata-on-user-messages),
and [Stubbing Custom Actions](https://rasa.com/docs/rasa-pro/production/testing-your-assistant#stubbing-custom-actions)
known from end-to-end (E2E) tests are supported as well.
They behave the same as in E2E tests.

### Explanation of the Commands

[Commands](https://rasa.com/docs/rasa-pro/concepts/dialogue-understanding#command-reference)
are the core components that direct the chatbot's actions in response to a user's input.
Each command represents an operation or decision the bot should make.
Here are the default commands that can be used in a test case:

- **StartFlow(flow_name)**: Starts a new flow with the specified flow name.
- **CancelFlow()**: Cancels the current flow.
- **SetSlot(slot_name, slot_value)**: Assigns a value to a specific slot.
- **Clarify(options)**: Seeks clarification by presenting options to the user.
  The options are optional in Dialogue Understanding Tests.
- **ChitChat()**: Initiates a casual conversation or response.
- **SearchAndReply()**: Performs a search operation and generates a reply.
- **HumanHandoff()**: Transfers the conversation to a human agent.
- **SkipQuestion()**: The user asked to skip a certain step.
- **RepeatLastBotMessages()**: Repeats the last bot message(s) to the user.

The syntax of the commands matches the domain specific language (DSL) used in the prompt templates of the
command generators.

*Note*: It is also possible to use custom commands in the test cases
(see section "Evaluation of Custom Commands").

### Explanation of `placeholder_generated_answer`

The `placeholder_generated_answer` is used in scenarios where a bot response is dynamically generated, such as
when the bot retrieves information from an external knowledge base.
In such cases, you may not know the exact wording of the bot's response ahead of time.
This placeholder should be used in the test case where a specific bot response is expected but may vary due
to external dynamic content or search results.
It signals that the exact bot utterance is not fixed, yet the test case recognizes and accepts a dynamically
generated response in its place.

Here is an example test case that uses `placeholder_generated_answer`:

```yaml
test_cases:
  - test_case: user asks a knowledge question during flow
    steps:
      - user: I want to send some money to Tre
        commands:
          - StartFlow(transfer_money)
          - SetSlot(transfer_money_recipient, Tre)
      - utter: utter_ask_transfer_money_amount_of_money
      - user: btw, are these transfers free of charge?
        commands:
          - SearchAndReply()
      - utter: placeholder_generated_answer
      - utter: utter_ask_transfer_money_amount_of_money
      - user: great, 50$ then
        commands:
          - SetSlot(transfer_money_amount_of_money, 50)
      - utter: utter_ask_transfer_money_final_confirmation
      - user: yes
        commands:
          - SetSlot(transfer_money_final_confirmation, True)
      - utter: utter_transfer_complete
```

## Evaluation of Custom Commands

### Defining new Custom Commands

To evaluate custom commands in Dialogue Understanding Tests, you need to define the custom command
as a subclass of `Command` and implement the `to_dsl`, `from_dsl`, and `regex_pattern` methods.

Here is an example of a custom command:

```python
from rasa.dialogue_understanding.commands import Command


class TestCommand(Command):
    ...

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        return "Test()"

    @classmethod
    def from_dsl(cls, dsl: str) -> "TestCommand":
        # Parse the DSL string and create a CustomCommand object
        return TestCommand()

    @staticmethod
    def regex_pattern() -> str:
        # Define the regex pattern that matches the DSL string
        return r"Test\(\)"
```

After defining the custom command, you can instruct the command parser to parse this new custom command
from your custom command generator by passing the custom command as an additional command in your
`parse_commands` method:

```python
@classmethod
def parse_commands(
        cls, actions: Optional[str], tracker: DialogueStateTracker, flows: FlowsList
) -> List[Command]:
    """Parse the actions returned by the llm into intent and entities.

    Args:
        actions: The actions returned by the llm.
        tracker: The tracker containing the current state of the conversation.
        flows: the list of flows

    Returns:
        The parsed commands.
    """
    from rasa.dialogue_understanding.generator.command_parser import (
        parse_commands as parse_commands_using_command_parsers,
    )

    return parse_commands_using_command_parsers(actions, flows, additional_commands=[TestCommand])
```

The additional commands are passed as a list of custom command classes.

When running the Dialogue Understanding Tests, you can pass the custom command as a cli argument:

```bash
rasa test du <path-to-test-cases> --additional-commands my_module.TestCommand
```

The `--additional-commands` argument takes a list of custom command classes separated by spaces.

### Updating Default Commands

If you want to update the default commands, you can subclass the default command and override any of the methods.

In this example, we update the regex of the `CancelFlow` command to match the DSL string `Cancel()` instead of
`CancelFlow()`:

```python
from rasa.dialogue_understanding.commands import CancelFlow


class CustomCancelFlow(CancelFlow):
    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        return "Cancel()"

    @staticmethod
    def regex_pattern() -> str:
        # Define the regex pattern that matches the DSL string
        return r"Cancel\(\)"
```

After defining the custom command, you can instruct the command parser to parse this new custom command
from your command generator by passing the updated default command as an additional command and
removing the default command:

```python
@classmethod
def parse_commands(
        cls, actions: Optional[str], tracker: DialogueStateTracker, flows: FlowsList
) -> List[Command]:
    """Parse the actions returned by the llm into intent and entities.

    Args:
        actions: The actions returned by the llm.
        tracker: The tracker containing the current state of the conversation.
        flows: the list of flows

    Returns:
        The parsed commands.
    """
    from rasa.dialogue_understanding.generator.command_parser import (
        parse_commands as parse_commands_using_command_parsers,
    )

    return parse_commands_using_command_parsers(actions, flows, additional_commands=[CustomCancelFlow],
                                                remove_default_commands=[CancelFlow])
```

When running the Dialogue Understanding Tests, you can pass the updated default command as a cli argument:

```bash
rasa test du <path-to-test-cases> --additional-commands my_module.CustomCancelFlow --remove-default-commands CancelFlow
```

Like the `--additional-commands` arg, the `--remove-default-commands` arg takes a list of default command classes
separated by spaces.

**Note**: The class name alone is sufficient for the `--remove-default-commands` argument because the default commands
are
already known by the Dialogue Understanding Test framework.

## Criteria for Test Case Success

A test case is considered to have passed if all the expected commands match the predicted commands at
each step. The expected and predicted commands are considered identical if their types and arguments
exactly match, with the order of the commands being irrelevant.
To compare two commands we use the `__eq__` method of the commands.
There's an exception for the `Clarify` command.

**Clarify Command**

When defining a `Clarify` command in a Dialogue Understanding Test, you have the option to leave the
command's options empty or specify a list of options; the options are optional.
If you provide a list of options, the predicted `Clarify` command must include the exact same list
to match the expected command. If you leave the options list empty, the predicted `Clarify` command
can have any list of options.

## Dialogue Understanding Test Output

The output of Dialogue Understanding Tests provides a comprehensive view of the chatbot's performance in
predicting and generating commands.
It includes detailed information that helps users pinpoint areas of improvement in the command generation process.
The output is logged to the console and saved in a detailed report file in a structured format for later analysis or
record-keeping.

The following information is present in the output:

- Number of passed and failed test cases.
- Number of passed and failed user utterances.
- Test case names of failed and passed test cases.
- A detailed diff of expected vs. predicted commands for each failed user message in a failed test case.
  A test case can have multiple failed user messages. The command generators listed in the output are the ones that
  generated the predicted commands.
  Example of a failed test case diff:
    ```diff
    ------------- test_case: <file-path>::user_adds_contact_to_their_list -------------

    Number of failed steps: 1

    == failure starting at user message 'it's @barts'.

    -- COMMAND GENERATOR(s) --
    SingleStepLLMCommandGenerator

    -- CONVERSATION --
    user: I want to add someone to my contact list
    bot: What's the handle of the user you want to add?
    user: it's @barts
    -- EXPECTED --          | -- PREDICTED --
    SetSlot(handle, @barts) | SetSlot(name, @barts)
    ```
- Command metrics for each command type, including the total count, true positives (tp), false positives (fp),
  false negatives (fn), precision, recall, and f1-score.
  Example of command metrics:
    ```diff
    start flow (total count: 10):
      tp: 10 fp: 0 fn: 0
      precision: 1.00
      recall   : 1.00
      f1       : 1.00
    ```

If you start the dialogue understanding tests with the `--output-prompt` flag, you will also see the prompt that
returned the predicted commands.

## Converting end-to-end (E2E) Tests to Dialogue Understanding Tests

To convert end-to-end (E2E) tests into Dialogue Understanding Tests you can use a standalone Python script:

```bash
python convert_e2e_tests_to_du_tests.py <path-to-e2e-tests>
```

The script has the following parameters:

- `<path-to-e2e-tests>`: The path to your existing E2E test cases (can be a single file or a directory).
- `--output-folder <output-folder>`: The path where the converted test cases will be saved. The default is
  `dialogue_understanding_tests`.

After running the script, the output folder structure will look like this:

```bash
<output-folder>
|-- ready
|  |-- test_case_a.yml
|  |-- test_case_b.yml
|  |-- ...
|-- to_review
|  |-- test_case_c.yml
|  |-- test_case_d.yml
|  |-- ...
```

Test cases that end up in **ready** are converted from E2E test cases that passed.
No further action is needed for these cases.
Test cases in **to_review** may require manual intervention because the E2E test failed.
Review these cases to ensure that the converted test cases are correct and the list of commands and
bot responses is complete.


## Converting DUT test from one DSL to a another DSL

If you need to transform your commands from one DSL format to another
(for instance, updating `StartFlow(flow_name)` to `start flow_name` or `SetSlot(slot_name, slot_value)` to `set slot_name slot_value`),
you can use a standalone Python script:

```bash
python convert_dut_dsl.py --dut-tests-dir <path> --output-dir <path> --dsl-mappings <path>
```

The script has the following required parameters:

- `--dut-tests-dir <path>`: The directory (relative or absolute) containing your
  existing Dialogue Understanding Tests (DUT). The script will look for `.yaml` or 
  `.yml` files within this folder (and subfolders).
- `--output-dir <path>`: The directory where transformed files will be saved. The folder
  structure from your `dut-tests-dir` is preserved.
- `--dsl-mappings <path>`: The YAML file defining your DSL mapping rules.

The YAML file containing the mappings must adhere to the following format: 
  ```yaml
  mappings:
  
  - from_dsl_regex: "^StartFlow\\(([^)]*)\\)$"
    to_dsl_pattern: "start {1}"
  
  - from_dsl_regex: "^SetSlot\\(([^,]+),\\s*(.*)\\)$"
    to_dsl_pattern: "set {1} {2}"
  
  - from_dsl_regex: "Clarify\(([\"\'a-zA-Z0-9_, ]*)\)"
    to_dsl_pattern: "clarify {1}"
    input_separators:
      - ","
      - " "
    output_separator: " "
  
  # ... add more mappings here

  ```

- `from_dsl_regex`: A regular expression (string) used to match the old DSL command.
  Must include any necessary anchors (like ^ and $) and capturing groups ( ... ) for 
  dynamic parts.
- `to_dsl_pattern`: A string that contains placeholders like `{1}`, `{2}`, etc. Each
  placeholder corresponds to a capturing group in from_dsl_regex, in order of
  appearance.
- `input_separators`: Optional list of separators of the captured groups that can be replaced
  with the `output_separator`
- `output_separator`: Output separator to replace separators from the list of `input_separators` in the captured group.
