#!/bin/bash

set -Eeuo pipefail

curl -sL https://sentry.io/get-cli/ | bash

# Assumes you're in a git repository
# envs that need to be set: `VERSION`, `SENTRY_ORG` and `SENTRY_AUTH_TOKEN`

# Create a release
sentry-cli releases new -p rasa-open-source "rasa-$VERSION"

# Associate commits with the release
sentry-cli releases set-commits --auto "rasa-$VERSION"

# once you are done, finalize
sentry-cli releases finalize "rasa-$VERSION"
