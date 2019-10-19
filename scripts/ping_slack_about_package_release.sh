#!/bin/bash

set -Eeuo pipefail

if [[ ${TRAVIS_TAG} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    curl -X POST -H "Content-type: application/json" \
	 --data "{\"text\":\"💥 New *Rasa* version ${TRAVIS_TAG} has been released! Changelog: https://github.com/RasaHQ/rasa/blob/${TRAVIS_TAG}/CHANGELOG.rst\"}" \
	 "https://hooks.slack.com/services/T0GHWFTS8/BMTQQL47K/${SLACK_WEBHOOK_TOKEN}"
fi

