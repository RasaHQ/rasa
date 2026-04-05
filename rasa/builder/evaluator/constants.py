"""Default literal values for evaluator configuration (env var fallbacks).

These are the defaults used when corresponding environment variables are unset.
Resolved runtime values live in ``rasa.builder.evaluator.config``.
"""

# Concurrency (overall evaluator pipeline when judges run locally)
DEFAULT_MAX_CONCURRENT_EVALUATIONS = 10

# Shared default model when a component-specific model env var is unset
DEFAULT_EVALUATOR_MODEL = "gpt-4.1-2025-04-14"

# Claim extractor
DEFAULT_CLAIM_EXTRACTOR_TEMPERATURE = 0.0
DEFAULT_CLAIM_EXTRACTOR_TIMEOUT = 100
DEFAULT_CLAIM_EXTRACTOR_MAX_TOKENS = 2000

# Faithfulness judge
DEFAULT_FAITHFULNESS_JUDGE_TEMPERATURE = 0.0
DEFAULT_FAITHFULNESS_JUDGE_TIMEOUT = 120
DEFAULT_FAITHFULNESS_JUDGE_MAX_TOKENS = 4000

# Completeness judge
DEFAULT_COMPLETENESS_JUDGE_TEMPERATURE = 0.0
DEFAULT_COMPLETENESS_JUDGE_TIMEOUT = 120
DEFAULT_COMPLETENESS_JUDGE_MAX_TOKENS = 4000
