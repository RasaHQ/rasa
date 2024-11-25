from rasa.shared.core.domain import Domain
from typing import Any, List, Optional, Dict, Text

from rasa.core.turns.utils.multi_label_encoder import MultiLabelEncoder
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.utils import bilou_utils
from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    NO_ENTITY_TAG,
    TEXT,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow import model_data_utils


class EntityTagsEncoder:
    def __init__(self, domain: Domain, bilou_tagging: bool = False) -> None:
        multi_label_encoder = MultiLabelEncoder(domain.entity_states)

        if bilou_tagging:
            tag_id_index_mapping = {
                f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
                for tag, idx_1 in multi_label_encoder.dimension_name_to_index.items()
                for idx_2, prefix in enumerate(BILOU_PREFIXES)
            }
        else:
            tag_id_index_mapping = {
                tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
                for tag, idx in multi_label_encoder.dimension_name_to_index.items()
            }

        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_index_mapping[NO_ENTITY_TAG] = 0

        # The entity states used to create the tag-idx-mapping contains the
        # entities and the concatenated entity and roles/groups.
        self.entity_tag_spec = EntityTagSpec(
            tag_name=ENTITY_ATTRIBUTE_TYPE,
            tags_to_ids=tag_id_index_mapping,
            ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
            num_tags=len(tag_id_index_mapping),
        )
        self.bilou_tagging = bilou_tagging

    def encode(
        self,
        text_tokens: Optional[List[Token]],
        entities: Optional[Dict[Text, Any]],
    ) -> List[Features]:

        if not entities or not text_tokens:
            return []

        dummy_message = Message()
        dummy_message.add(TOKENS_NAMES[TEXT], text_tokens)
        dummy_message.add(ENTITIES, entities)

        if self.bilou_tagging:
            bilou_utils.apply_bilou_schema_to_message(dummy_message)

        return [
            model_data_utils.get_tag_ids(
                dummy_message, self.entity_tag_spec, self.bilou_tagging
            )
        ]
