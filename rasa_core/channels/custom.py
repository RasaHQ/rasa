from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
from flask import Blueprint, request, jsonify

from rasa_core.channels.channel import UserMessage, OutputChannel, InputChannel
from rasa_core.channels import CollectingOutputChannel

logger = logging.getLogger(__name__)



