# EKS and integration-tests scripts

This file documents scripts under **infrastructure/eks-helm** and **infrastructure/eks-integration-tests**. Scripts in **eks-integration-tests/scripts** are used for RDS bootstrap, IRSA role creation, and (for eks-helm) manual stack destroy.

**RDS IAM auth (no password):** Ensure the RDS instance exists (e.g. from `eks-integration-tests`), IAM database authentication is enabled on the instance, and the DB user `integration_tests_user_no_password` exists with `rds_iam` (see `setup-databases.sh`). Then run `create-irsa-role.sh`; it will attach the `rds-db:connect` policy for that instance. Set the printed role ARN as GitHub secret `RASA_IRSA_ROLE_ARN`. Keep `Pulumi.ci.yaml` with `trackerStoreDbUser: "integration_tests_user_no_password"`. After deploy, Rasa uses `IAM_CLOUD_PROVIDER=aws` and `RDS_SQL_DB_AWS_IAM_ENABLED=true` to generate an IAM token and connect without a password.
This is a one off setup and is currently deployed.
---

## eks-integration-tests stack outputs

The **eks-integration-tests** (project name `integration-tests`) stack exports:

| Output | Description |
|--------|-------------|
| `db_host` | RDS endpoint hostname |
| `db_port` | Database port (5432) |
| `db_name` | Database name |
| `db_username` | Database username |
| `db_password` | Master password (stored as secret; use `pulumi stack output db_password --show-secrets` to decrypt) |
| `db_connection_string` | Full PostgreSQL connection string |

---

### infrastructure/eks-integration-tests/scripts/destroy-stack.sh

Destroys an **eks-helm** stack (not the integration-tests stack). Use when cleaning up an ephemeral eks-helm deployment manually (e.g. when the helm-deploy workflow cleanup step is skipped). Sets dummy eks-helm config so validation passes, then runs `pulumi destroy` and removes the stack.

**Prerequisites:** PULUMI_ACCESS_TOKEN set (or `pulumi login`), AWS credentials. Run from the **eks-helm** project directory so the correct Pulumi project is used; the script lives under eks-integration-tests but is intended for eks-helm. Copy the script to `infrastructure/eks-helm/scripts/` and run from there, or run the equivalent steps from `infrastructure/eks-helm`.

**Usage (from repo root, if using script from eks-helm):**

```bash
cd infrastructure/eks-helm
./scripts/destroy-stack.sh <STACK_NAME>
# Example:
./scripts/destroy-stack.sh vo-443-20260206143026
```

**Manual steps (from infrastructure/eks-helm):**

```bash
cd infrastructure/eks-helm
pulumi stack select rasa/eks-helm/<STACK_NAME>

# Dummy config (required for validation; not used during destroy)
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

pulumi destroy --yes
yes "rasa/eks-helm/<STACK_NAME>" | pulumi stack rm "rasa/eks-helm/<STACK_NAME>" --force --yes
```

### infrastructure/eks-integration-tests/scripts/create-irsa-role.sh

Creates the IRSA (IAM Roles for Service Accounts) IAM role **once** locally. The same role ARN is then used for every `infrastructure/eks-helm` deployment by setting it in the GitHub secret `RASA_IRSA_ROLE_ARN`.

**What it does**

- Reads the EKS cluster OIDC provider ARN and URL from the Pulumi stack `rasa/eks-cluster/ci`
- Creates or updates the IAM role `rasa-pro-ci-irsa` with a trust policy that allows the Rasa service account in any `rasa-pro-*` namespace to assume the role
- **RDS IAM auth:** If the RDS instance exists and is reachable, attaches an inline policy `RdsIamAuth` granting `rds-db:connect` for `arn:aws:rds-db:REGION:ACCOUNT:dbuser:DBI_RESOURCE_ID/integration_tests_user_no_password` so the Rasa pod can connect to PostgreSQL with IAM (no password)
- Prints the role ARN for you to set as the GitHub secret

**Prerequisites**

- Pulumi CLI logged in (`pulumi login` or `PULUMI_ACCESS_TOKEN` set)
- AWS CLI configured (credentials with permission to create/update IAM roles and describe RDS)
- Stack `rasa/eks-cluster/ci` must exist and export `oidcProviderArn` and `oidcProviderUrl`
- For RDS IAM policy: RDS instance must exist (e.g. from `eks-integration-tests` stack) and IAM auth enabled on the instance

**Usage**

From repo root:

```bash
./infrastructure/eks-integration-tests/scripts/create-irsa-role.sh
```

From `infrastructure/eks-integration-tests`:

```bash
./scripts/create-irsa-role.sh
```

Optional env vars:

| Env var | Default | Description |
|--------|---------|-------------|
| `IRSA_ROLE_NAME` | `rasa-pro-ci-irsa` | IAM role name |
| `RDS_DB_INSTANCE_ID` | `rasa-pro-integration-tests-db` | RDS instance identifier (used to fetch DbiResourceId and attach RDS IAM policy) |
| `RDS_DB_USERNAME` | `integration_tests_user_no_password` | IAM DB user in the policy resource ARN |
| `AWS_REGION` | `eu-west-1` | Region for RDS (and for policy resource ARN) |

Example with custom RDS instance:

```bash
RDS_DB_INSTANCE_ID=my-db AWS_REGION=eu-west-1 ./scripts/create-irsa-role.sh
```

**After running**

1. **Set the GitHub secret** `RASA_IRSA_ROLE_ARN` to the printed role ARN (e.g. `arn:aws:iam::329710836760:role/rasa-pro-ci-irsa`) in the repo’s **Settings → Secrets and variables → Actions**. (already added, only update if new RDS created)
2. The helm-deploy workflow uses this secret to set `eks-helm:irsaRoleArn` for each deployment.
3. **Other policies** (e.g. S3 for model bucket): attach with `aws iam attach-role-policy --role-name rasa-pro-ci-irsa --policy-arn <POLICY_ARN>`.


### infrastructure/eks-integration-tests/scripts/setup-databases.sh

Bootstraps the RDS instance created by the **eks-integration-tests** Pulumi project. Run it **after** the RDS instance is created. It connects as the RDS master user (e.g. `integration_tests_user`) and creates the assistant database/user and the RDS IAM auth user used by eks-helm (`integration_tests_user_no_password`).

**What it does**

- Creates the assistant user `DB_ASSISTANT_USERNAME`, grants `rds_iam`, creates the database `DB_ASSISTANT_DATABASE`, and sets ownership.
- Creates the user `integration_tests_user_no_password` and grants `rds_iam` (used by eks-helm as `trackerStoreDbUser` for RDS IAM auth).

**Required environment variables**

| Variable | Description |
|----------|-------------|
| `PGHOST` | RDS endpoint hostname |
| `PGUSER` | RDS master username (e.g. `integration_tests_user`) |
| `PGPASSWORD` | RDS master password |
| `DB_ASSISTANT_USERNAME` | Name of the assistant database user to create |
| `DB_ASSISTANT_DATABASE` | Name of the assistant database to create |

**Optional:** `PGPORT` (default `5432`), `PGDATABASE` (default `postgres`).

**Prerequisites**

- `psql` (PostgreSQL client) installed and on `PATH`.
- Network access to RDS (e.g. from within the VPC or via a jumphost).

**Usage**

From repo root, set env from the **eks-integration-tests** stack (project name `integration-tests`). `db_password` is a secret—use `--show-secrets` (or `-s`) to decrypt. `DB_ASSISTANT_USERNAME` and `DB_ASSISTANT_DATABASE` are not stack outputs; set them to the desired names or use the defaults below.

```bash
cd infrastructure/eks-integration-tests
pulumi stack select rasa/integration-tests/ci   # or your stack name

# Connection (from stack outputs; use --show-secrets for db_password)
export PGHOST=$(pulumi stack output db_host)
export PGPORT=$(pulumi stack output db_port)
export PGUSER=$(pulumi stack output db_username)
export PGPASSWORD=$(pulumi stack output db_password --show-secrets)
export PGDATABASE=postgres

# Assistant user and database (choose names that don't conflict with existing DBs)
export DB_ASSISTANT_USERNAME="${DB_ASSISTANT_USERNAME:-integration_tests_assistant_no_password}"
export DB_ASSISTANT_DATABASE="${DB_ASSISTANT_DATABASE:-assistant_db}"

./scripts/setup-databases.sh
```

The script enforces SSL via `PGSSLMODE=require` for all connections. See also `infrastructure/eks-integration-tests/README.md` for full RDS and bootstrap documentation.

