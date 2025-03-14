from rasa.core.nlg.interpolator import (
    RASA_FORMAT_TEMPLATE_ENGINE,
    _get_variables_to_be_rendered,
    interpolate,
    interpolate_format_template,
)


async def test_nlg_interpolator_get_variables_to_be_rendered() -> None:
    response = "Hello {name}, how are you?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert _get_variables_to_be_rendered(response, values) == {"name": "Rasa"}


async def test_nlg_interpolator_get_variables_to_be_rendered_no_variables() -> None:
    response = "Hello, how are you?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert _get_variables_to_be_rendered(response, values) == {}


async def test_nlg_interpolator_get_variables_to_be_rendered_empty_response() -> None:
    response = ""
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert _get_variables_to_be_rendered(response, values) == {}


async def test_nlg_interpolator_get_variables_to_be_rendered_no_values() -> None:
    response = "Hello {name}, how are you?"
    values = {}
    assert _get_variables_to_be_rendered(response, values) == {}


async def test_nlg_interpolator_get_variables_to_be_rendered_many_variables() -> None:
    response = "Hello {name}, how are you in {city}?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert _get_variables_to_be_rendered(response, values) == {
        "name": "Rasa",
        "city": "Berlin",
    }


async def test_nlg_interpolator_get_variables_to_be_rendered_empty_braces() -> None:
    response = "Hello {name}, {}?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert _get_variables_to_be_rendered(response, values) == {"name": "Rasa"}


async def test_nlg_interpolator_interpolate_format_template() -> None:
    response = "Hello {name}, how are you?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert interpolate_format_template(response, values) == "Hello Rasa, how are you?"


async def test_nlg_interpolator_interpolate_format_template_no_variables() -> None:
    response = "Hello, how are you?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert interpolate_format_template(response, values) == "Hello, how are you?"


async def test_nlg_interpolator_interpolate_format_template_empty_response() -> None:
    response = ""
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert interpolate_format_template(response, values) == ""


async def test_nlg_interpolator_interpolate_format_template_no_values() -> None:
    response = "Hello {name}, how are you?"
    values = {}
    assert interpolate_format_template(response, values) == "Hello {name}, how are you?"


async def test_nlg_interpolator_interpolate_format_template_many_variables() -> None:
    response = "Hello {name}, how are you in {city}?"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert (
        interpolate_format_template(response, values)
        == "Hello Rasa, how are you in Berlin?"
    )


async def test_nlg_interpolator_interpolate_format_template_empty_braces() -> None:
    response = "Hello {}"
    values = {"name": "Rasa", "age": "100", "city": "Berlin"}
    assert interpolate_format_template(response, values) == "Hello {}"


async def test_nlg_interpolator_interpolate_translation() -> None:
    response = {
        "text": "Add {add_contact_name} to your contacts?",
        "translation": {
            "it": "Aggiungi {add_contact_name} ai tuoi contatti?",
            "de": "Fügen Sie {add_contact_name} zu Ihren Kontakten hinzu?",
        },
    }
    values = {"add_contact_name": "Rasa", "add_contact_handle": "rasa"}
    interpolated_response = interpolate(response, values, RASA_FORMAT_TEMPLATE_ENGINE)
    expected_response = {
        "text": "Add Rasa to your contacts?",
        "translation": {
            "it": "Aggiungi Rasa ai tuoi contatti?",
            "de": "Fügen Sie Rasa zu Ihren Kontakten hinzu?",
        },
    }
    assert interpolated_response == expected_response
