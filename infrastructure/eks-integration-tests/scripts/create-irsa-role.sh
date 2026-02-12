#!/usr/bin/env bash
# Create the IRSA IAM role once (locally). NOTE: This is a one off manual setup.
# Use the same role ARN for every infrastructure/eks-helm deployment by setting it in the RASA_IRSA_ROLE_ARN
# GitHub secret (and optionally in Pulumi config).

# Prerequisites: pulumi CLI logged in, AWS CLI configured, infra dir.
# Usage: from repo root: ./infrastructure/eks-helm/scripts/create-irsa-role.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EKS_CLUSTER_DIR="$(cd "$SCRIPT_DIR/../../eks-cluster" && pwd)"
IRSA_ROLE_NAME="${IRSA_ROLE_NAME:-rasa-pro-ci-irsa}"

echo "Using IRSA role name: $IRSA_ROLE_NAME"
echo "Reading OIDC outputs from stack rasa/eks-cluster/ci..."
cd "$EKS_CLUSTER_DIR"
pulumi stack select rasa/eks-cluster/ci
OIDC_ARN=$(pulumi stack output oidcProviderArn)
OIDC_URL=$(pulumi stack output oidcProviderUrl)
OIDC_HOST="${OIDC_URL#https://}"

TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "$OIDC_ARN" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringLike": { "$OIDC_HOST:sub": "system:serviceaccount:rasa-pro-*:rasa" }
    }
  }]
}
EOF
)

echo "Creating or updating IAM role..."
if aws iam create-role --role-name "$IRSA_ROLE_NAME" --assume-role-policy-document "$TRUST_POLICY" 2>/dev/null; then
  echo "Role created."
else
  aws iam update-assume-role-policy --role-name "$IRSA_ROLE_NAME" --policy-document "$TRUST_POLICY"
  echo "Role trust policy updated."
fi

# Optional: attach RDS IAM auth policy so the role can connect to RDS with IAM (no password).
# Set RDS_DB_INSTANCE_ID to your RDS instance identifier (default: rasa-pro-integration-tests-db).
# Set RDS_DB_USERNAME to the IAM DB user (default: integration_tests_user_no_password).
RDS_DB_INSTANCE_ID="${RDS_DB_INSTANCE_ID:-rasa-pro-integration-tests-db}"
RDS_DB_USERNAME="${RDS_DB_USERNAME:-integration_tests_user_no_password}"
RDS_REGION="${AWS_REGION:-eu-west-1}"

if aws rds describe-db-instances --db-instance-identifier "$RDS_DB_INSTANCE_ID" --region "$RDS_REGION" --query 'DBInstances[0].DbiResourceId' --output text 2>/dev/null | grep -q .; then
  DBI_RESOURCE_ID=$(aws rds describe-db-instances --db-instance-identifier "$RDS_DB_INSTANCE_ID" --region "$RDS_REGION" --query 'DBInstances[0].DbiResourceId' --output text)
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  RDS_POLICY_RESOURCE="arn:aws:rds-db:${RDS_REGION}:${ACCOUNT_ID}:dbuser:${DBI_RESOURCE_ID}/${RDS_DB_USERNAME}"
  RDS_POLICY=$(cat <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "rds-db:connect",
    "Resource": "$RDS_POLICY_RESOURCE"
  }]
}
POLICY
)
  aws iam put-role-policy --role-name "$IRSA_ROLE_NAME" --policy-name RdsIamAuth --policy-document "$RDS_POLICY"
  echo "Attached inline policy RdsIamAuth (rds-db:connect) for $RDS_POLICY_RESOURCE"
else
  echo "RDS instance '$RDS_DB_INSTANCE_ID' not found or no access; skipping RDS IAM policy."
  echo "To attach RDS IAM auth later, run:"
  echo "  DBI_RESOURCE_ID=\$(aws rds describe-db-instances --db-instance-identifier $RDS_DB_INSTANCE_ID --region $RDS_REGION --query 'DBInstances[0].DbiResourceId' --output text)"
  echo "  ACCOUNT_ID=\$(aws sts get-caller-identity --query Account --output text)"
  echo "  # Then create and attach a policy with Resource: arn:aws:rds-db:$RDS_REGION:\$ACCOUNT_ID:dbuser:\$DBI_RESOURCE_ID/$RDS_DB_USERNAME"
fi

# S3: allow Rasa to read models from the trained-models bucket (HeadBucket, ListBucket, GetObject).
S3_BUCKET_NAME="${S3_BUCKET_NAME:-rasa-pro-ci-trained-models}"
S3_POLICY=$(cat <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::${S3_BUCKET_NAME}"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:HeadObject"],
      "Resource": "arn:aws:s3:::${S3_BUCKET_NAME}/*"
    }
  ]
}
POLICY
)
aws iam put-role-policy --role-name "$IRSA_ROLE_NAME" --policy-name S3ModelBucket --policy-document "$S3_POLICY"
echo "Attached inline policy S3ModelBucket for bucket $S3_BUCKET_NAME"

IRSA_ROLE_ARN=$(aws iam get-role --role-name "$IRSA_ROLE_NAME" --query 'Role.Arn' --output text)
echo ""
echo "IRSA role ARN (set this as GitHub secret RASA_IRSA_ROLE_ARN and use for every eks-helm deployment):"
echo "$IRSA_ROLE_ARN"
