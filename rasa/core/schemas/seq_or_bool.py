# -*- coding: utf-8 -*-
import logging

log = logging.getLogger(__name__)


def ext_str(value, rule_obj, path):
    log.debug("value: %s", value)
    log.debug("rule_obj: %s", rule_obj)
    log.debug("path: %s", path)

    # Either raise some exception that you have defined your self
    # raise AssertionError('Custom assertion error in jinja_function()')

    # Or you should return True/False that will tell if it validated
    return True


def ext_list(value, rule_obj, path):
    log.debug("value: %s", value)
    log.debug("rule_obj: %s", rule_obj)
    log.debug("path: %s", path)

    # Either raise some exception that you have defined your self
    # raise AssertionError('Custom assertion error in jinja_function()')

    # Or you should return True/False that will tell if it validated
    return True


def ext_map(value, rule_obj, path):
    log.debug("value: %s", value)
    log.debug("rule_obj: %s", rule_obj)
    log.debug("path: %s", path)

    # Either raise some exception that you have defined your self
    # raise AssertionError('Custom assertion error in jinja_function()')

    # Or you should return True/False that will tell if it validated
    return True
