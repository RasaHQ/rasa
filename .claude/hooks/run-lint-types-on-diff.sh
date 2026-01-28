#!/usr/bin/env bash
set -euo pipefail

repo_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$repo_dir"

diff="$(git diff HEAD)"
if [[ -z "$diff" ]]; then
  exit 0
fi

diff_hash="$(printf "%s" "$diff" | shasum -a 256 | cut -d ' ' -f1)"
state_file="$(git rev-parse --git-dir)/claude-lint-types-last"

if [[ -f "$state_file" ]]; then
  last_hash="$(cat "$state_file")"
  if [[ "$diff_hash" == "$last_hash" ]]; then
    exit 0
  fi
fi

set +e
make lint
lint_status=$?
make types
types_status=$?
set -e

if [[ $lint_status -ne 0 || $types_status -ne 0 ]]; then
  exit 2
fi

printf "%s" "$diff_hash" > "$state_file"
