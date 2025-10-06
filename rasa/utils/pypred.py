"""Wrapper module for pypred that provides a fixed Predicate class.

This module should be used instead of importing directly from pypred.

This patch fixes an issue where pypred creates excessive logs of being unable
to write a file when run in an environment with no write access is given.

https://rasahq.atlassian.net/browse/ATO-1925

The solution is based on https://github.com/FreeCAD/FreeCAD/issues/6315

This patch also prevents duplicate "yacc table file version is out of date"
warnings by caching the PLY parser after first initialization. Previously, each
Predicate object creation would trigger a new PLY parser initialization,
so the same warning was printed multiple times throughout training.

https://rasahq.atlassian.net/browse/ENG-2392
"""

from typing import Any

import ply.yacc
import pypred.parser
from pypred import Predicate as OriginalPredicate  # noqa: TID251

# Store the original yacc function
_original_yacc = ply.yacc.yacc


# Cache the PLY parser to prevent multiple initializations
_cached_parser = None


def patched_yacc(*args: Any, **kwargs: Any) -> Any:
    global _cached_parser

    # Return cached parser if available
    if _cached_parser is not None:
        return _cached_parser

    # Disable generation of debug ('parser.out') and table
    # cache ('parsetab.py'), as it requires a writable location.
    kwargs["write_tables"] = False
    kwargs["module"] = pypred.parser

    # Create parser once and cache it
    _cached_parser = _original_yacc(*args, **kwargs)
    return _cached_parser


# Apply the patch
ply.yacc.yacc = patched_yacc


class Predicate(OriginalPredicate):
    """Fixed version of pypred.Predicate that uses the patched yacc parser."""

    pass
