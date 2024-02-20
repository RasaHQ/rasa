from typing import List, Optional, Text

import pycountry
from rasa.shared.exceptions import RasaException

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRule,
    AnonymizationRuleList,
)
from rasa.anonymization.utils import (
    read_endpoint_config,
    validate_anonymization_yaml,
)

KEY_ANONYMIZATION_RULES = "anonymization"


class AnonymizationRulesYamlReader:
    """Reads anonymization rules in YAML."""

    def __init__(self, endpoints_filename: Optional[Text] = None) -> None:
        """Initializes the reader with the endpoints' filename."""
        self.endpoints_filename = endpoints_filename

    def read_anonymization_rules(self) -> List[AnonymizationRuleList]:
        """Reads Anonymization rules from a YAML file.

        Returns:
            Parsed Anonymization rules.
        """
        yaml_content = read_endpoint_config(
            self.endpoints_filename, KEY_ANONYMIZATION_RULES
        )
        if yaml_content is None:
            return []

        validate_anonymization_yaml(yaml_content)

        anonymization_rules = []

        for key, value in yaml_content.items():
            if key == KEY_ANONYMIZATION_RULES:
                metadata = value.get("metadata", {})
                rule_lists = value.get("rule_lists", [])

                lang_code = metadata.get("language")
                self.validate_language(lang_code)

                model_provider = metadata.get("model_provider")
                model_name = metadata.get("model_name")

                for rule in rule_lists:
                    identifier = rule.get("id")
                    rules = rule.get("rules", [])
                    rule_list = []

                    for item in rules:
                        entity_name = item.get("entity")
                        substitution = item.get("substitution", "mask")
                        value = item.get("value")

                        rule_obj = AnonymizationRule(
                            entity_name=entity_name,
                            substitution=substitution,
                            value=value,
                        )
                        rule_list.append(rule_obj)
                    anonymization_rule_list_obj = AnonymizationRuleList(
                        id=identifier,
                        rule_list=rule_list,
                        language=lang_code,
                        model_provider=model_provider,
                        models=model_name,
                    )

                    anonymization_rules.append(anonymization_rule_list_obj)

        return anonymization_rules

    def validate_language(self, lang_code: Text) -> None:
        """Checks if the language is a valid ISO 639-2 code."""
        language = pycountry.languages.get(alpha_2=lang_code)
        if language is None:
            raise RasaException(
                f"Provided language code '{lang_code}' is invalid. "
                f"In order to proceed with anonymization, "
                f"please provide a valid ISO 639-2 language code in "
                f"the {self.endpoints_filename} file."
            )

        return None
