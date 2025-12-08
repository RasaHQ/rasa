"""Constants for the copilot response evaluator."""

# Experiment configuration
EXPERIMENT_NAME = "Copilot Response Quality Evaluation"
EXPERIMENT_DESCRIPTION = (
    "Evaluating Copilot response quality on faithfulness and completeness dimensions. "
    "Faithfulness measures whether claims in responses are grounded in evidence. "
    "Completeness measures whether responses fully address all parts of user requests."
)

# Faithfulness metric names
FAITHFULNESS_SUPPORT_RATE_METRIC = "faithfulness_support_rate"
FAITHFULNESS_AVG_VERDICT_CONFIDENCE_METRIC = "faithfulness_avg_verdict_confidence"
FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_METRIC = "faithfulness_supported_claims_count"
FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_METRIC = "faithfulness_contradicted_claims_count"
FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_METRIC = (
    "faithfulness_not_enough_info_claims_count"
)
FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_METRIC = "faithfulness_extracted_claims_count"
FAITHFULNESS_REASONING_METADATA_METRIC = "faithfulness_reasoning_metadata"
FAITHFULNESS_REASONING_METADATA_DESCRIPTION = (
    "Reasoning data for faithfulness evaluation."
)

# Completeness metric names
COMPLETENESS_COVERED_PARTS_METRIC = "completeness_covered_parts"
COMPLETENESS_UNCOVERED_PARTS_METRIC = "completeness_uncovered_parts"
COMPLETENESS_TOTAL_PARTS_METRIC = "completeness_total_parts"
COMPLETENESS_COVERAGE_RATE_METRIC = "completeness_coverage_rate"
COMPLETENESS_CONFIDENCE_METRIC = "completeness_confidence"
COMPLETENESS_REASONING_METADATA_METRIC = "completeness_reasoning_metadata"
COMPLETENESS_REASONING_METADATA_DESCRIPTION = (
    "Reasoning data for completeness evaluation."
)
# Failure metric names
FAITHFULNESS_FAILURE_METRIC = "faithfulness_evaluation_failed"
COMPLETENESS_FAILURE_METRIC = "completeness_evaluation_failed"

# Faithfulness metric descriptions
FAITHFULNESS_SUPPORT_RATE_DESCRIPTION = (
    "Faithfulness support rate (supported claims/total claims): {value:.3f}"
)
FAITHFULNESS_AVG_VERDICT_CONFIDENCE_DESCRIPTION = (
    "Average confidence across all claim verdicts: {value:.3f}"
)
FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_DESCRIPTION = (
    "Number of claims with supported verdicts: {value}"
)
FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_DESCRIPTION = (
    "Number of claims with contradicted verdicts: {value}"
)
FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_DESCRIPTION = (
    "Number of claims with not enough info verdicts: {value}"
)
FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_DESCRIPTION = (
    "Count of atomic claims extracted from copilot response: {value}"
)


# Completeness metric descriptions
COMPLETENESS_COVERED_PARTS_DESCRIPTION = "Number of user request parts covered: {value}"
COMPLETENESS_UNCOVERED_PARTS_DESCRIPTION = (
    "Number of user request parts not covered: {value}"
)
COMPLETENESS_TOTAL_PARTS_DESCRIPTION = "Total parts in user request: {value}"
COMPLETENESS_COVERAGE_RATE_DESCRIPTION = (
    "Completeness coverage rate (covered/total): {value:.3f}"
)
COMPLETENESS_CONFIDENCE_DESCRIPTION = "Completeness assessment confidence: {value:.3f}"

# Failure metric descriptions
FAITHFULNESS_FAILURE_DESCRIPTION = (
    "Faithfulness evaluation failed: {error_type} - {error_message}"
)
COMPLETENESS_FAILURE_DESCRIPTION = (
    "Completeness evaluation failed: {error_type} - {error_message}"
)
