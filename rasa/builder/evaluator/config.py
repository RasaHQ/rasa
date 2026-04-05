import os

from rasa.builder.evaluator.constants import (
    DEFAULT_CLAIM_EXTRACTOR_MAX_TOKENS,
    DEFAULT_CLAIM_EXTRACTOR_TEMPERATURE,
    DEFAULT_CLAIM_EXTRACTOR_TIMEOUT,
    DEFAULT_COMPLETENESS_JUDGE_MAX_TOKENS,
    DEFAULT_COMPLETENESS_JUDGE_TEMPERATURE,
    DEFAULT_COMPLETENESS_JUDGE_TIMEOUT,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_FAITHFULNESS_JUDGE_MAX_TOKENS,
    DEFAULT_FAITHFULNESS_JUDGE_TEMPERATURE,
    DEFAULT_FAITHFULNESS_JUDGE_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_EVALUATIONS,
)

# Concurrency Configuration
# This controls the overall concurrency limit for the evaluator pipeline when we run
# the judges ourselves. Langfuse runners orchestrate concurrency on their side, so
# these values do not meaningfully affect Langfuse-powered experiments.
MAX_CONCURRENT_EVALUATIONS = int(
    os.getenv("MAX_CONCURRENT_EVALUATIONS", str(DEFAULT_MAX_CONCURRENT_EVALUATIONS))
)

# Claim Extractor Configuration
CLAIM_EXTRACTOR_MODEL = os.getenv("CLAIM_EXTRACTOR_MODEL", DEFAULT_EVALUATOR_MODEL)
CLAIM_EXTRACTOR_TEMPERATURE = float(
    os.getenv("CLAIM_EXTRACTOR_TEMPERATURE", str(DEFAULT_CLAIM_EXTRACTOR_TEMPERATURE))
)
CLAIM_EXTRACTOR_TIMEOUT = int(
    os.getenv("CLAIM_EXTRACTOR_TIMEOUT", str(DEFAULT_CLAIM_EXTRACTOR_TIMEOUT))
)
CLAIM_EXTRACTOR_MAX_TOKENS = int(
    os.getenv("CLAIM_EXTRACTOR_MAX_TOKENS", str(DEFAULT_CLAIM_EXTRACTOR_MAX_TOKENS))
)
# Claim extraction processes all responses first, so it uses the full concurrency limit.
# Langfuse runners orchestrate concurrency on their side, so these values do not
# meaningfully affect Langfuse-powered experiments.
CLAIM_EXTRACTOR_MAX_CONCURRENT_EXTRACTIONS = int(
    os.getenv(
        "CLAIM_EXTRACTOR_MAX_CONCURRENT_EXTRACTIONS", str(MAX_CONCURRENT_EVALUATIONS)
    )
)

# Faithfulness Judge Configuration
FAITHFULNESS_JUDGE_MODEL = os.getenv(
    "FAITHFULNESS_JUDGE_MODEL", DEFAULT_EVALUATOR_MODEL
)
FAITHFULNESS_JUDGE_TEMPERATURE = float(
    os.getenv(
        "FAITHFULNESS_JUDGE_TEMPERATURE", str(DEFAULT_FAITHFULNESS_JUDGE_TEMPERATURE)
    )
)
FAITHFULNESS_JUDGE_TIMEOUT = int(
    os.getenv("FAITHFULNESS_JUDGE_TIMEOUT", str(DEFAULT_FAITHFULNESS_JUDGE_TIMEOUT))
)
FAITHFULNESS_JUDGE_MAX_TOKENS = int(
    os.getenv(
        "FAITHFULNESS_JUDGE_MAX_TOKENS", str(DEFAULT_FAITHFULNESS_JUDGE_MAX_TOKENS)
    )
)
# Faithfulness and completeness judges run in parallel per entry, so each gets half
# the concurrency limit to avoid exceeding the overall target. Langfuse runners
# orchestrate concurrency on their side, so these values do not meaningfully affect
# Langfuse-powered experiments.
FAITHFULNESS_JUDGE_MAX_CONCURRENT_EVALUATIONS = int(
    os.getenv(
        "FAITHFULNESS_JUDGE_MAX_CONCURRENT_EVALUATIONS",
        str(max(1, round(MAX_CONCURRENT_EVALUATIONS / 2))),
    )
)

# Completeness Judge Configuration
COMPLETENESS_JUDGE_MODEL = os.getenv(
    "COMPLETENESS_JUDGE_MODEL", DEFAULT_EVALUATOR_MODEL
)
COMPLETENESS_JUDGE_TEMPERATURE = float(
    os.getenv(
        "COMPLETENESS_JUDGE_TEMPERATURE", str(DEFAULT_COMPLETENESS_JUDGE_TEMPERATURE)
    )
)
COMPLETENESS_JUDGE_TIMEOUT = int(
    os.getenv("COMPLETENESS_JUDGE_TIMEOUT", str(DEFAULT_COMPLETENESS_JUDGE_TIMEOUT))
)
COMPLETENESS_JUDGE_MAX_TOKENS = int(
    os.getenv(
        "COMPLETENESS_JUDGE_MAX_TOKENS", str(DEFAULT_COMPLETENESS_JUDGE_MAX_TOKENS)
    )
)
# Completeness judge gets the same concurrency as faithfulness since they run
# concurrently. Langfuse runners orchestrate concurrency on their side, so these values
# do not meaningfully affect Langfuse-powered experiments.
COMPLETENESS_JUDGE_MAX_CONCURRENT_EVALUATIONS = int(
    os.getenv(
        "COMPLETENESS_JUDGE_MAX_CONCURRENT_EVALUATIONS",
        str(max(1, round(MAX_CONCURRENT_EVALUATIONS / 2))),
    )
)
