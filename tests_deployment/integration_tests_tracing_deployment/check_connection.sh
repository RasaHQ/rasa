#!/usr/bin/env bash

max_iterations=15
wait_seconds=8

iterations=0

while true; do
  ((iterations++))
  echo "Attempt $iterations to reach $1"

  http_code=$(curl --verbose -s -o /dev/null -w '%{http_code}' "$1")
  if [ "$http_code" -eq 200 ]; then
    echo "Returning 200 from $1"
    exit 0
  fi

  if [ "$iterations" -ge "$max_iterations" ]; then
    echo "Loop Timeout"
    exit 1
  fi

  sleep $wait_seconds
done
