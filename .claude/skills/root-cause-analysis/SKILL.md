---
name: root-cause-analysis
description: Diagnoses the most likely root cause of CI failures by correlating logs with code and recent changes.
---
# Root Cause Analysis

## When to use
Use this skill after summarizing CI failures to determine the most plausible cause.

## Instructions
1. Map error messages to code locations when possible.
2. Distinguish between test failures, infra issues, and product code defects.
3. Provide the minimal causal chain (symptom -> cause -> fix target).

## Output format
- Root cause hypothesis
- Supporting evidence
- Confidence (low/medium/high)

## Mode
ultrathink
