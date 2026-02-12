#!/usr/bin/env bash
# Destroy an eks-helm stack without needing real config.
# Sets dummy config so validation passes; values are not used during destroy.
#
# Usage: ./scripts/destroy-stack.sh [STACK_NAME]
# Example: ./scripts/destroy-stack.sh vo-443-20260206143026
# From repo root: ./infrastructure/eks-integration-tests/scripts/destroy-stack.sh vo-443-20260206143026

# NOTE: This is done in the helm-workflow to clean up the resources after the tests are run. 
# This script is only used to clean up the resources manually after a deployment needs to be tested or 
# some tests need to be run on it and the helm-workflow step is therefore skipped.
set -euo pipefail

STACK_NAME="${1:?Usage: $0 STACK_NAME (e.g. vo-443-20260206143026)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives in infrastructure/eks-integration-tests/scripts/; eks-helm is sibling under infrastructure/
EKS_HELM_DIR="$(cd "$SCRIPT_DIR/../../eks-helm" && pwd)"

cd "$EKS_HELM_DIR"

echo "Selecting stack rasa/eks-helm/$STACK_NAME..."
pulumi stack select "rasa/eks-helm/$STACK_NAME"

echo "Setting dummy config (required for validation; not used during destroy)..."
pulumi config set eks-helm:clusterName "rasa/eks-cluster/ci"
pulumi config set eks-helm:projectName "dummy"
pulumi config set eks-helm:rasaPro "dummy"
pulumi config set eks-helm:rasaProRepository "dummy"
pulumi config set eks-helm:rasaProHelmRepo "dummy"
pulumi config set eks-helm:rasaProHelmVersion "dummy"
pulumi config set eks-helm:rasaProModelBucketName "dummy"
pulumi config set eks-helm:rasaProModelFileName "dummy"
pulumi config set eks-helm:trackerStoreDbHost "dummy"
pulumi config set eks-helm:trackerStoreDbPort "5432"
pulumi config set eks-helm:trackerStoreDbName "dummy"
pulumi config set eks-helm:trackerStoreDbUser "dummy"
pulumi config set eks-helm:irsaRoleArn "arn:aws:iam::000000000000:role/dummy"

echo "Running pulumi destroy..."
pulumi destroy -s "rasa/eks-helm/$STACK_NAME" --yes

echo "Removing stack..."
yes "rasa/eks-helm/$STACK_NAME" | pulumi stack rm "rasa/eks-helm/$STACK_NAME" --force --yes

echo "Done."
