from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

import jsonpickle

from rasa_core.conversation import Topic
from rasa_core.domain import TemplateDomain
from rasa_core.trackers import DialogueStateTracker, TopicSet


def tracker_from_dialogue_file(filename, domain=None):
    dialogue = read_dialogue_file(filename)

    dialogue_topics = set([Topic(t.topic)
                           for t in dialogue.events
                           if isinstance(t, TopicSet)])
    if domain is not None:
        domain = domain
    else:
        domain = TemplateDomain.load("examples/default_domain.yml")
    domain.topics.extend(dialogue_topics)
    tracker = DialogueStateTracker(dialogue.name, domain.slots,
                                   domain.topics, domain.default_topic)
    tracker.recreate_from_dialogue(dialogue)
    return tracker


def read_dialogue_file(filename):
    with io.open(filename, "r") as f:
        dialogue_json = f.read()
    return jsonpickle.loads(dialogue_json)
