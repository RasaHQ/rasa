"""``rasa tools`` CLI command family.

Subcommands:
- ``rasa tools init`` — interactive setup wizard
- ``rasa tools init skills`` — fetch and install agent skills
- ``rasa tools init docs`` — fetch offline documentation
- ``rasa tools run`` — start the MCP server
"""

from rasa.cli.tools.parsers import add_subparser

__all__ = ["add_subparser"]
