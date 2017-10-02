from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core import utils
from rasa_core.conversation import Topic, DefaultTopic

dummy_topics = [Topic("topic_{0}".format(i)) for i in range(4)]


def test_default():
    stack = utils.TopicStack(dummy_topics, [], DefaultTopic)
    assert stack.top.name == DefaultTopic.name, \
        "default topic not correctly set"


def test_push_pop():
    stack = utils.TopicStack(dummy_topics, [], DefaultTopic)
    stack.push(dummy_topics[0])
    assert stack.top.name == dummy_topics[0].name, \
        "pushing a topic should put that topic at the top"
    stack.push(dummy_topics[0])
    assert stack.top.name == dummy_topics[0].name, \
        "pushing a topic a second time should have no effect"
    stack.pop()
    assert len(stack) == 0, \
        "popping the last topic should leave the stack empty"
    assert stack.top.name == DefaultTopic.name, \
        "popping a the last topic should leave default on top"
    stack.pop()
    assert len(stack) == 0, \
        "popping an empty stack should leave it unchanged"


def test_push_multiple():
    stack = utils.TopicStack(dummy_topics, [], DefaultTopic)
    stack.push(dummy_topics[0])
    stack.push(dummy_topics[1])
    stack.push(dummy_topics[2])
    assert stack.top.name == dummy_topics[2].name, \
        "last pushed topic should be on top"
    stack.push(dummy_topics[1])
    assert stack.top.name == dummy_topics[1].name, \
        "last pushed topic should be on top"
    assert len([t for t in stack if
                t.name == dummy_topics[1].name]) == 1, \
        "pushing multiple times shouldn't produce duplicates"
    stack.pop()
    assert stack.top.name == dummy_topics[2].name, \
        "popping should revert to previous top"
    stack.pop()
    assert stack.top.name == dummy_topics[0].name, \
        "should be no duplicates in the stack"
