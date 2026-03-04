#!/bin/bash
# Wrapper script for running pytest with pytest-split
# Handles exit code 5 (no tests collected) gracefully for empty split groups

set -o pipefail

MAKE_TARGET="${1:?Make target is required}"
RUNNER_ID="${RUNNER_ID:-unknown}"

# Capture both stdout/stderr and exit code
# We need to check the output because make wraps pytest's exit code
output=$(make "${MAKE_TARGET}" 2>&1) || exit_code=$?
echo "$output"

# Exit code 5 means no tests collected - treat as success for empty splits
# This can happen when pytest-split assigns 0 tests to a runner group
# Note: make returns exit code 2 when a recipe fails, so we check for both
# and also verify the output contains the pytest "Error 5" message
if [ "${exit_code:-0}" != "0" ]; then
    if echo "$output" | grep -q "Error 5$" || [ "${exit_code}" = "5" ]; then
        echo "::warning::No tests in this split group (runner ${RUNNER_ID}) - this is expected when test count < runner count"
        exit 0
    fi
fi

exit ${exit_code:-0}
