from rasa.dialogue_understanding.commands import ClarifyCommand, StartFlowCommand
from rasa.dialogue_understanding_test.command_comparison import (
    _are_clarify_commands_equal,
    are_command_lists_equal,
    are_commands_equal,
    is_command_present_in_list,
)


class TestCommandComparison:
    # Tests for are_command_lists_equal
    def test_empty_lists_equal(self):
        assert are_command_lists_equal([], []) is True

    def test_lists_different_lengths(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")
        assert are_command_lists_equal([cmd1], [cmd1, cmd2]) is False
        assert are_command_lists_equal([cmd1, cmd2], [cmd1]) is False

    def test_lists_same_length_different_commands(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")
        assert are_command_lists_equal([cmd1], [cmd2]) is False

    def test_lists_same_commands_different_order(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")
        assert are_command_lists_equal([cmd1, cmd2], [cmd2, cmd1]) is True

    # Edge case: Lists with duplicate commands
    def test_lists_with_duplicates(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow1")
        assert are_command_lists_equal([cmd1, cmd1], [cmd1, cmd2]) is True

    # Edge case: Very large lists
    def test_large_command_lists(self):
        large_list1 = [StartFlowCommand("flow1") for _ in range(1000)]
        large_list2 = [StartFlowCommand("flow1") for _ in range(1000)]
        assert are_command_lists_equal(large_list1, large_list2) is True

    # Tests for is_command_present_in_list
    def test_command_present_single_item(self):
        cmd = StartFlowCommand("flow")
        assert is_command_present_in_list(cmd, [cmd]) is True

    def test_command_not_present(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")
        assert is_command_present_in_list(cmd1, [cmd2]) is False

    # Edge case: Empty list
    def test_command_present_empty_list(self):
        cmd = StartFlowCommand("flow1")
        assert is_command_present_in_list(cmd, []) is False

    # Edge case: List with many similar commands
    def test_command_present_many_similar(self):
        target_cmd = StartFlowCommand("flow1")
        similar_cmds = [StartFlowCommand("flow2") for _ in range(100)]
        assert is_command_present_in_list(target_cmd, similar_cmds) is False

    # Tests for are_commands_equal
    def test_regular_commands_equal(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow1")
        assert are_commands_equal(cmd1, cmd2) is True

    def test_regular_commands_not_equal(self):
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")
        assert are_commands_equal(cmd1, cmd2) is False

    # Edge cases for ClarifyCommand comparisons
    def test_clarify_commands_both_with_options(self):
        cmd1 = ClarifyCommand(options=["opt1", "opt2"])
        cmd2 = ClarifyCommand(options=["opt1", "opt2"])
        assert are_commands_equal(cmd1, cmd2) is True

    def test_clarify_commands_different_options(self):
        cmd1 = ClarifyCommand(options=["opt1"])
        cmd2 = ClarifyCommand(options=["opt2"])
        assert are_commands_equal(cmd1, cmd2) is False

    def test_clarify_commands_one_without_options(self):
        cmd1 = ClarifyCommand(options=[])
        cmd2 = ClarifyCommand(options=["opt1"])
        assert are_commands_equal(cmd1, cmd2) is True

    def test_clarify_commands_both_without_options(self):
        cmd1 = ClarifyCommand(options=[])
        cmd2 = ClarifyCommand(options=[])
        assert are_commands_equal(cmd1, cmd2) is True

    # Edge case: Mixed command types
    def test_mixed_command_types(self):
        regular_cmd = StartFlowCommand("flow")
        clarify_cmd = ClarifyCommand(options=[])
        assert are_commands_equal(regular_cmd, clarify_cmd) is False

    # Tests for _are_clarify_commands_equal
    def test_clarify_equal_exact_match(self):
        cmd1 = ClarifyCommand(options=["opt1", "opt2"])
        cmd2 = ClarifyCommand(options=["opt1", "opt2"])
        assert _are_clarify_commands_equal(cmd1, cmd2) is True

    def test_clarify_equal_different_order_of_options(self):
        cmd1 = ClarifyCommand(options=["opt1", "opt2"])
        cmd2 = ClarifyCommand(options=["opt2", "opt1"])
        assert _are_clarify_commands_equal(cmd1, cmd2) is True

    def test_clarify_not_equal_different_options(self):
        cmd1 = ClarifyCommand(options=["opt1"])
        cmd2 = ClarifyCommand(options=["opt2"])
        assert _are_clarify_commands_equal(cmd1, cmd2) is False

    # Edge case: Empty options lists
    def test_clarify_empty_options(self):
        cmd1 = ClarifyCommand(options=[])
        cmd2 = ClarifyCommand(options=[])
        assert _are_clarify_commands_equal(cmd1, cmd2) is True

    # Edge case: None vs empty options
    def test_clarify_empty_vs_one_options(self):
        cmd1 = ClarifyCommand(options=[])
        cmd2 = ClarifyCommand(options=["opt1"])
        assert _are_clarify_commands_equal(cmd1, cmd2) is True
