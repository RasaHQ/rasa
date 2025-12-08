import os

# Concurrency Configuration
# This controls the overall concurrency limit for the evaluator pipeline when we run
# the judges ourselves. Langfuse runners orchestrate concurrency on their side, so
# these values do not meaningfully affect Langfuse-powered experiments.
MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "10"))

# Claim Extractor Configuration
CLAIM_EXTRACTOR_MODEL = os.getenv("CLAIM_EXTRACTOR_MODEL", "gpt-4.1-2025-04-14")
CLAIM_EXTRACTOR_TEMPERATURE = float(os.getenv("CLAIM_EXTRACTOR_TEMPERATURE", "0"))
CLAIM_EXTRACTOR_TIMEOUT = int(os.getenv("CLAIM_EXTRACTOR_TIMEOUT", "100"))
CLAIM_EXTRACTOR_MAX_TOKENS = int(os.getenv("CLAIM_EXTRACTOR_MAX_TOKENS", "2000"))
# Claim extraction processes all responses first, so it uses the full concurrency limit.
# Langfuse runners orchestrate concurrency on their side, so these values do not
# meaningfully affect Langfuse-powered experiments.
CLAIM_EXTRACTOR_MAX_CONCURRENT_EXTRACTIONS = int(
    os.getenv(
        "CLAIM_EXTRACTOR_MAX_CONCURRENT_EXTRACTIONS", str(MAX_CONCURRENT_EVALUATIONS)
    )
)

# Faithfulness Judge Configuration
FAITHFULNESS_JUDGE_MODEL = os.getenv("FAITHFULNESS_JUDGE_MODEL", "gpt-4.1-2025-04-14")
FAITHFULNESS_JUDGE_TEMPERATURE = float(os.getenv("FAITHFULNESS_JUDGE_TEMPERATURE", "0"))
FAITHFULNESS_JUDGE_TIMEOUT = int(os.getenv("FAITHFULNESS_JUDGE_TIMEOUT", "120"))
FAITHFULNESS_JUDGE_MAX_TOKENS = int(os.getenv("FAITHFULNESS_JUDGE_MAX_TOKENS", "4000"))
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
COMPLETENESS_JUDGE_MODEL = os.getenv("COMPLETENESS_JUDGE_MODEL", "gpt-4.1-2025-04-14")
COMPLETENESS_JUDGE_TEMPERATURE = float(os.getenv("COMPLETENESS_JUDGE_TEMPERATURE", "0"))
COMPLETENESS_JUDGE_TIMEOUT = int(os.getenv("COMPLETENESS_JUDGE_TIMEOUT", "120"))
COMPLETENESS_JUDGE_MAX_TOKENS = int(os.getenv("COMPLETENESS_JUDGE_MAX_TOKENS", "4000"))
# Completeness judge gets the same concurrency as faithfulness since they run
# concurrently. Langfuse runners orchestrate concurrency on their side, so these values
# do not meaningfully affect Langfuse-powered experiments.
COMPLETENESS_JUDGE_MAX_CONCURRENT_EVALUATIONS = int(
    os.getenv(
        "COMPLETENESS_JUDGE_MAX_CONCURRENT_EVALUATIONS",
        str(max(1, round(MAX_CONCURRENT_EVALUATIONS / 2))),
    )
)
