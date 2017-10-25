# -*- coding: utf-8 -*-
"""Python part of the warnings subsystem."""

# Note: function level imports should *not* be used
# in this module as it may cause import lock deadlock.
# See bug 683658.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import linecache
import sys
import warnings

import six


def py2_monkey_patch_warnings():
    if six.PY2:

        def formatwarning(message, category, filename, lineno, line=None):
            """Function to format a warning the standard way."""
            s = "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)
            line = linecache.getline(filename, lineno) if line is None else line
            if line:
                line = line.strip()
                s += "  %s\n" % line.decode("utf-8")
            return s

        sys.modules['warnings'].formatwarning = formatwarning
