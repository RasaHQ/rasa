from typing import Any

from core.policies.ted_policy import TEDPolicy


class KnowledgeBasePolicy(TEDPolicy):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    # TODO
