"""Wrapper module for pypred that provides a fixed Predicate class.

This module should be used instead of importing directly from pypred.

This patch fixes an issue where pypred creates excessive logs of being unable
to write a file when run in an environment with no write access is given.

https://rasahq.atlassian.net/browse/ATO-1925

The solution is based on https://github.com/FreeCAD/FreeCAD/issues/6315
"""

import logging
from typing import Any

import ply.yacc
import pypred.parser
from pypred import Predicate as OriginalPredicate  # noqa: TID251

# Store the original yacc function
_original_yacc = ply.yacc.yacc

# Create a logger that suppresses warnings to avoid yacc table file version warnings
_yacc_logger = logging.getLogger("ply.yacc")
_yacc_logger.setLevel(logging.ERROR)


def patched_yacc(*args: Any, **kwargs: Any) -> Any:
    # Disable generation of debug ('parser.out') and table
    # cache ('parsetab.py'), as it requires a writable location.
    kwargs["write_tables"] = False
    kwargs["module"] = pypred.parser
    # Suppress yacc warnings by using a logger that only shows errors
    kwargs["errorlog"] = _yacc_logger
    return _original_yacc(*args, **kwargs)


# Apply the patch
ply.yacc.yacc = patched_yacc


class Predicate(OriginalPredicate):
    """Fixed version of pypred.Predicate that uses the patched yacc parser."""

    pass
