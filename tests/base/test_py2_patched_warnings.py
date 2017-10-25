# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import pytest

import rasa_nlu.py2_patched_warnings as patched_warnings


def test_standard_warnings():
    """
    standard library `warnings` under python2 can't handle unicode correctly,
    it will raise an exception
    """
    with pytest.raises(UnicodeEncodeError):
        warnings.warn("中文")


def test_patched_warnings():
    """
    patched `warnings` can handle unicode correctly
    """

    patched_warnings.warn("中文")
    patched_warnings.warn("english")
