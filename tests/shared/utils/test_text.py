import pytest

from rasa.shared.utils.text import contains_ssml_tags


@pytest.mark.parametrize(
    "text",
    [
        '<prosody rate="slow">Hello</prosody>',
        "Say <break time='500ms'/> something",
        '<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme>',
        "<emphasis level='strong'>important</emphasis>",
        '<say-as interpret-as="cardinal">42</say-as>',
        '<sub alias="World Wide Web">WWW</sub>',
        "<mark name='bookmark1'/>",
        "<s>A sentence</s>",
        "<p>A paragraph</p>",
        '<lang xml:lang="fr-FR">Bonjour</lang>',
        '<audio src="beep.wav"/>',
        "<bookmark mark='test'/>",
    ],
)
def test_ssml_detection_positive(text):
    assert contains_ssml_tags(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Hello, how are you?",
        "5 > 3 and 2 < 4",
        "Use <speak> tag to wrap",
        "<voice name='en-US'>Hello</voice>",
        "No tags here at all",
        "",
        "angle brackets < > in math",
    ],
)
def test_ssml_detection_negative(text):
    assert contains_ssml_tags(text) is False
