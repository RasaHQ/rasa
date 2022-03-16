import logging
from typing import Any, Dict, List, Text, Optional

from rasa.shared.core.slots import Slot

logger = logging.getLogger(__name__)


class LimitSlot(Slot):
    """
    A slot for featurizing an amount as greater than or equal to vs. less than a given value.

    Example of configuration in the domain.yml file:
    slots:
        my_slot:
          type: custom.slots.LimitSlot
          limit: 100
    """

    type_name = "limit"

    def __init__(
        self,
        name: Text,
        limit: int,
        mappings: List[Dict[Text, Any]],
        initial_value: Any = None,
        value_reset_delay: Optional[int] = None,
        influence_conversation: bool = True,
    ) -> None:

        super().__init__(
            name=name,
            initial_value=initial_value,
            mappings=mappings,
            value_reset_delay=value_reset_delay,
            influence_conversation=influence_conversation,
        )
        self.limit = limit

    def _as_feature(self) -> List[float]:
        try:
            greater_than_limit = float(self.value >= self.limit)
            return [1.0, greater_than_limit]
        except (TypeError, ValueError):
            return [0.0, 0.0]

    def persistence_info(self) -> Dict[Text, Any]:
        """Returns relevant information to persist this slot."""
        d = super().persistence_info()
        d["limit"] = self.limit
        return d

    def _feature_dimensionality(self) -> int:
        return len(self.as_feature())
