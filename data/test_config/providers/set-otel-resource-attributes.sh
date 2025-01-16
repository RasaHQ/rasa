#!/usr/bin/env bash
set -eu pipefail

RASA_CALM_DEMO_BRANCH="${RASA_CALM_DEMO_BRANCH:=main}"
RASA_PRO_VERSION="$(rasa --version | grep 'Rasa Pro Version' | tr -d ' ' | awk -F ':' '{print $2}')"
RASA_PRIVATE_BRANCH="$(git branch --show-current)"
RASA_PRIVATE_SHA="$(git rev-parse --short HEAD)"

echo "RASA_CALM_DEMO_BRANCH=${RASA_CALM_DEMO_BRANCH}"
echo "RASA_PRO_VERSION=${RASA_PRO_VERSION}"
echo "RASA_PRIVATE_BRANCH=${RASA_PRIVATE_BRANCH}"
echo "RASA_PRIVATE_SHA=${RASA_PRIVATE_SHA}"

OTEL_RESOURCE_ATTRIBUTES=rasa-pro-version="${RASA_PRO_VERSION}",rasa-private-branch="${RASA_PRIVATE_BRANCH}",rasa-private-sha="${RASA_PRIVATE_SHA}",rasa-calm-demo-branch="${RASA_CALM_DEMO_BRANCH}"
echo "OTEL_RESOURCE_ATTRIBUTES=${OTEL_RESOURCE_ATTRIBUTES}"

export RASA_CALM_DEMO_BRANCH
export RASA_PRO_VERSION
export RASA_PRIVATE_BRANCH
export RASA_PRIVATE_SHA
export OTEL_RESOURCE_ATTRIBUTES
