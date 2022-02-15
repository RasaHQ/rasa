import argparse
from typing import List, Optional, Text, Dict

# TODO: In addition to read_file and read_yaml, use validate_yaml_schema once
# we have schema ready for validation.
from rasa.shared.utils.io import read_file, read_yaml
from rasa.shared.exceptions import FileNotFoundException


class ProjectConfig:
    """Class which manages project structure declaration.

    Main file here is `project.yml`. CLI arguments have the highest priority,
    and legacy importers in `config.yml` are used as a fallback. If legacy project
    is missing `project.yml` file, some defaults and heuristics are used to
    fill configuration variables so that legacy execution continues
    consistently. Heuristics here mean checking files in data directory and
    sorting files according to their type, ie. NLU/stories/rules etc.
    """

    def __init__(self, args: Optional[argparse.Namespace] = None):
        """Create a ProjectConfig instance object.

        Create an instance from `project.yml`, CLI args, defaults and
        `config.yml` (only for importer configuration).

        Args:
            args: Namespace arguments.
        """
        self.read_project_file()
        self.set_cli_override(args)
        self.set_training_files()
        self.set_importer_fallback()
        self.set_empty_with_defaults()

    def read_project_file(self):
        """Read project file and fill config.

        Legacy projects do not have a project file, and this is not a
        show-stopper; but if the file is there, all keys are mandatory (apart
        from importers) and validation ensures that.
        """
        try:
            self.file_content = read_file("project.yml")
        except FileNotFoundException:
            self.file_content = None
            self.config = dict()
            return
        # TODO
        # validate_yaml_schema(self.file_content, project_schema)
        self.config = read_yaml(self.file_content)

    def set_cli_override(self, args):
        """Override project config with CLI arguments."""
        if getattr(args, "config"):
            # FIXME: This can be a list, not a single value, for
            # `rasa train core` and `rasa test nlu`.
            self.config["config"] = args.config
        # TODO: For model I think we need a proper method.
        # "model" or "out + fixed_model_name" as used by `rasa train`?
        # or model as positional argument for `rasa shell`, `rasa interactive`
        # and `rasa run` (positional argument is in addition to --model)
        cli_override_keys = ["config", "domain", "model", "stories", "nlu"]
        # TODO: We need another method for `--data`? Or set_training_files will
        # fix this? If yes, set_training_files needs args too.

    def set_training_files(self):
        """If `project.yml` is missing, check data dir to find training files.

        This method does the same heuristics as pre Rasa 3.1 to be backwards
        compatible, ie. when a legacy project does not have `projecy.yml` file.
        This can also be used when migration command is creating the new
        `project.yml` file.
        """

    def set_importer_fallback(self):
        """Fallback for importer using config.yml file."""
        if self.config["importers"]:
            # If we already have importers in project.yml, we don't need the
            # legacy config.yml because of priority.
            return

    def set_empty_with_defaults(self):
        """Fills missing keys with defaults if `project.yml` is missing."""

    def __getitem__(self, key):
        """Method for dictionary like access (ie. square brackets) of ProjectConfig."""
