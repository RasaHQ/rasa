---
name: infra-flake-detection
description: Detects CI failures caused by infrastructure or environment instability (timeouts, network, resource limits).
---
# Infra Flake Detection

## When to use
Use when logs show timeouts, connection resets, retries, or resource exhaustion.

## Instructions
1. Look for non-deterministic failures across steps.
2. Identify hints like "timeout", "ECONNRESET", "Killed", "OOM", "No space".
3. Recommend stabilizing changes (timeouts, retries, caching, resource limits).

## Output format
- Infra symptom
- Likely cause
- Stabilization change

## Mode
ultrathink
