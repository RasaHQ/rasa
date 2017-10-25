# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import six


if six.PY2:
    from rasa_nlu.py2_patched_warnings import py2_monkey_patch_warnings
    py2_monkey_patch_warnings()


if six.PY2:
    def test_patched_warnings():
        """
        patched `warnings` can handle unicode correctly
        """

        warnings.warn("中文")
        warnings.warn("english")
