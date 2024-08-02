import logging
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Union

from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import (
    SpacyNlpEngine,
    StanzaNlpEngine,
    TransformersNlpEngine,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from rasa.shared.exceptions import RasaException

from rasa.utils.singleton import Singleton

if typing.TYPE_CHECKING:
    from presidio_analyzer.nlp_engine.nlp_engine import NlpEngine

DEFAULT_PRESIDIO_LANG_CODE = "en"
DEFAULT_PRESIDIO_MODEL_NAME = "en_core_web_lg"
DEFAULT_PRESIDIO_MODEL_PROVIDER = "spacy"

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationRule:
    """A rule for anonymizing a given text."""

    entity_name: Text
    substitution: Text = "mask"
    value: Optional[Text] = None


@dataclass
class AnonymizationRuleList:
    """A list of anonymization rules."""

    id: Text
    rule_list: List[AnonymizationRule]
    language: Text = DEFAULT_PRESIDIO_LANG_CODE
    model_provider: Text = DEFAULT_PRESIDIO_MODEL_PROVIDER
    models: Union[Text, Dict[Text, Text]] = DEFAULT_PRESIDIO_MODEL_NAME


class AnonymizationAnalyzer(metaclass=Singleton):
    """Anonymization analyzer."""

    presidio_analyzer_engine = None

    def __init__(self, anonymization_rule_list: AnonymizationRuleList):
        """Initialise the anonymization analyzer."""
        if self.presidio_analyzer_engine is None:
            self.presidio_analyzer_engine = self._get_analyzer_engine(
                anonymization_rule_list
            )

    @staticmethod
    def _get_analyzer_engine(
        anonymization_rule_list: AnonymizationRuleList,
    ) -> AnalyzerEngine:
        """Returns an analyzer engine for all the anonymization rule lists."""
        try:
            nlp_engine = AnonymizationAnalyzer._build_presidio_nlp_engine(
                anonymization_rule_list
            )
        except (OSError, ImportError) as exception:
            raise RasaException(
                "Failed to load Presidio nlp engine. "
                "Please check that you have provided "
                "a valid model name."
            ) from exception

        return AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[anonymization_rule_list.language],
        )

    @staticmethod
    def _build_presidio_nlp_engine(
        anonymization_rule_list: AnonymizationRuleList,
    ) -> "NlpEngine":
        """Creates an instance of the Presidio nlp engine."""
        if anonymization_rule_list.model_provider == "transformers":
            nlp_engine = TransformersNlpEngine(
                models={
                    anonymization_rule_list.language: anonymization_rule_list.models
                },
            )
        elif anonymization_rule_list.model_provider == "stanza":
            nlp_engine = StanzaNlpEngine(
                models={
                    anonymization_rule_list.language: anonymization_rule_list.models
                },
            )
        else:
            nlp_engine = SpacyNlpEngine(
                models={
                    anonymization_rule_list.language: anonymization_rule_list.models
                },
            )

        return nlp_engine


class AnonymizationRuleExecutor:
    """Executes a given anonymization rule set on a given text."""

    def __init__(self, anonymization_rule_list: AnonymizationRuleList):
        """Initialize the anonymization rule executor."""
        self.anonymization_rule_list = anonymization_rule_list

        is_valid_rule_list = self._validate_anonymization_rule_list(
            anonymization_rule_list
        )

        self.analyzer = (
            AnonymizationAnalyzer(anonymization_rule_list)
            if is_valid_rule_list
            else None
        )

        self.anonymizer_engine = AnonymizerEngine()  # type: ignore

    @staticmethod
    def _validate_anonymization_rule_list(
        anonymization_rule_list: AnonymizationRuleList,
    ) -> bool:
        """Validates the given anonymization rule list object."""
        if (
            anonymization_rule_list.language != DEFAULT_PRESIDIO_LANG_CODE
            and anonymization_rule_list.models == DEFAULT_PRESIDIO_MODEL_NAME
        ):
            logger.debug(
                f"Anonymization rule list language is "
                f"'{anonymization_rule_list.language}', "
                f"but no specific model name was provided. "
                f"You must specify the spaCy model name in the"
                f"endpoints yaml file. "
                f"Cannot proceed with anonymization."
            )
            return False

        return True

    def run(self, text: Text) -> Optional[Text]:
        """Anonymizes the given text using the given anonymization rule list."""
        if (
            self.analyzer is None
            or not self.anonymization_rule_list
            or self.analyzer.presidio_analyzer_engine is None
        ):
            return text

        if not self.anonymization_rule_list.rule_list:
            return text

        analyzer_results = self.analyzer.presidio_analyzer_engine.analyze(
            text=text,
            entities=[
                rule.entity_name for rule in self.anonymization_rule_list.rule_list
            ],
            language=self.anonymization_rule_list.language,
        )

        operators = self.get_operators()

        anonymized_text = self.anonymizer_engine.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators,
        )

        return anonymized_text.text

    @staticmethod
    @typing.no_type_check  # faker is not typed correctly
    def _get_supported_faker_entities() -> Dict[Text, Any]:
        faker = Faker(["en_US", "es_ES", "it_IT"])

        # Presidio entities: https://microsoft.github.io/presidio/supported_entities/
        # Faker providers: https://faker.readthedocs.io/en/master/providers.html
        # Unsupported entities by faker:
        # CRYPTO, NRP, MEDICAL_LICENSE, US_BANK_NUMBER, US_DRIVER_LICENSE
        # UK_NHS, IT_FISCAL_CODE, IT_DRIVER_LICENSE, IT_PASSPORT, IT_IDENTITY_CARD
        # SG_NRIC_FIN, AU_ABN, AU_ACN, AU_TFN, AU_MEDICARE
        supported_entities = {
            "PERSON": lambda value: faker["en_US"].name(),
            "PHONE_NUMBER": lambda value: faker["en_US"].phone_number(),
            "EMAIL_ADDRESS": lambda value: faker["en_US"].ascii_email(),
            "CREDIT_CARD": lambda value: faker["en_US"].credit_card_number(),
            "IBAN_CODE": lambda value: faker["en_US"].iban(),
            "DATE_TIME": lambda value: faker["en_US"].date(),
            "IP_ADDRESS": lambda value: faker["en_US"].ipv4(),
            "URL": lambda value: faker["en_US"].url(),
            # This faker method returns a tuple of
            # (latitude, longitude, place name, country code, timezone)
            "LOCATION": lambda value: faker["en_US"].location_on_land()[2],
            # USA
            "US_ITIN": lambda value: faker["en_US"].itin(),
            "US_PASSPORT": lambda value: faker["en_US"].passport_number(),
            "US_SSN": lambda value: faker["en_US"].ssn(),
            # Spain
            "ES_NIF": lambda value: faker["es_ES"].nif(),
            # Italy
            "IT_VAT_CODE": lambda value: faker["it_IT"].company_vat(),
        }

        return supported_entities

    @staticmethod
    def _mask_anonymize(value: Text) -> Text:
        return "*" * len(value)

    def get_substitution_func(self, rule: AnonymizationRule) -> Optional[Any]:
        """Returns a function that anonymizes the given text.

        Args:
            rule: The anonymization rule to use.

        Returns:
            A function that anonymizes the given text.
        """
        if rule.substitution == "faker":
            supported_faker_entities = self._get_supported_faker_entities()

            if rule.entity_name not in supported_faker_entities:
                logger.debug(
                    f"Unsupported faker entity: {rule.entity_name}. "
                    f"Supported entities are: {supported_faker_entities.keys()}"
                    f"Using mask anonymization instead."
                )
                func = self._mask_anonymize
            else:
                func = supported_faker_entities.get(rule.entity_name)
        elif rule.substitution == "mask":
            func = self._mask_anonymize
        else:
            raise RasaException(f"Unknown substitution type: {rule.substitution}")

        return func

    def get_operators(self) -> Dict[Text, OperatorConfig]:
        """Returns a dictionary of operators for the given anonymization rule list."""
        operators = {}

        for rule in self.anonymization_rule_list.rule_list:
            if rule.substitution == "text":
                operators[rule.entity_name] = OperatorConfig(
                    "replace", {"new_value": rule.value}
                )
            else:
                operators[rule.entity_name] = OperatorConfig(
                    "custom", {"lambda": self.get_substitution_func(rule)}
                )

        return operators
