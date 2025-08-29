from typing import Literal

LAKERA_API_KEY_ENV_VAR = "LAKERA_API_KEY"
LAKERA_GUARD_ENDPOINT = "guard"
LAKERA_GUARD_RESULTS_ENDPOINT = "guard/results"

BLOCK_SCOPE_USER: Literal["user"] = "user"
BLOCK_SCOPE_PROJECT: Literal["project"] = "project"
BlockScope = Literal["user", "project"]
