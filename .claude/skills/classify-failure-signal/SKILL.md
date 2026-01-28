---
name: classify-failure-signal
description: Classifies CI failures as false-positive or true-positive using evidence-based criteria.
---
# Classify Failure Signal

## When to use
Use when a decision is needed on whether failures indicate test noise or product defects.

## Instructions
1. Mark as false-positive if evidence points to flaky infra, test bugs, or bad fixtures/config.
2. Mark as true-positive if product code behavior violates expectations or specs.
3. State the strongest evidence for the classification.

## Output format
- Classification: false-positive | true-positive
- Evidence
- Confidence

## Mode
ultrathink
