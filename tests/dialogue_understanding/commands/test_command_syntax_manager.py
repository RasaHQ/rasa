import structlog

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)


class TestCommandSyntaxManager:
    def test_set_and_get_syntax_version(self) -> None:
        # Set the syntax version to v2.
        CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

        # Check if the syntax version is v2.
        assert CommandSyntaxManager.get_syntax_version() == CommandSyntaxVersion.v2

        # Reset the syntax version.
        CommandSyntaxManager.reset_syntax_version()

    def test_get_default_syntax_version(self) -> None:
        assert (
            CommandSyntaxManager.get_default_syntax_version() == CommandSyntaxVersion.v1
        )

    def test_set_syntax_version_already_set(self) -> None:
        # Set the syntax version to v2.
        CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

        # Try to set the syntax version again.
        with structlog.testing.capture_logs() as caplog:
            CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v1)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "command_syntax_manager.syntax_version_already_set":
                found_validation_log = True
                break

        # Check if the validation log was found.
        assert found_validation_log

        # Check if the syntax version is v1.
        assert CommandSyntaxManager.get_syntax_version() == CommandSyntaxVersion.v1

        # Reset the syntax version.
        CommandSyntaxManager.reset_syntax_version()

    def test_reset_syntax_version(self) -> None:
        # Reset the syntax version.
        CommandSyntaxManager.reset_syntax_version()

        # Check if the syntax version is None.
        assert CommandSyntaxManager.get_syntax_version() is None
