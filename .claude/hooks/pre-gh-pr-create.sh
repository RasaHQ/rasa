#!/usr/bin/env bash
set -euo pipefail
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
if [[ "$CMD" != *"git commit"* ]]; then
  exit 0  # Not a commit - allow
fi
# About to commit - ensure lint/types pass first
exec "${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/run-lint-types-on-diff.sh"
