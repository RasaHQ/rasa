# Integration Tests RDS Database with Pulumi

This project provisions an RDS PostgreSQL database for integration tests within an existing AWS EKS cluster VPC using Pulumi. The database is configured to be accessible only within the cluster's VPC for security.

## Overview

The Pulumi script performs the following steps:

1. **Network Configuration:**  
   Uses the existing EKS cluster VPC and private subnets from the base infrastructure stack.

2. **Database Subnet Group:**  
   Creates a DB subnet group using the private subnets from the EKS cluster.

3. **Security Group:**  
   Creates a security group allowing PostgreSQL access (port 5432) only from within the VPC CIDR (10.0.0.0/16).

4. **RDS Instance:**  
   Provisions a PostgreSQL 16.6 database with the following specifications:
   - Instance class: `db.t4g.small`
   - Storage: 50 GB gp3
   - Single-AZ deployment
   - Not publicly accessible
   - Database name: `integrationtestsdb`

5. **Secrets Management:**  
   Database credentials are managed through Pulumi's built-in secrets management system for secure access.

## Resources

A typical `pulumi up` will show resources like:

```
     Type                                                              Name                            Status
 +   pulumi:pulumi:Stack                                               eks-integration-tests-ci        created (5m30s)      
 +   ├─ aws:rds:SubnetGroup                                            rasa-pro-integration-tests-db-subnet-group created (2s)        
 +   ├─ aws:ec2:SecurityGroup                                          rasa-pro-integration-tests-db-sg created (1s)          
 +   ├─ aws:rds:Instance                                               rasa-pro-integration-tests-db   created (5m15s)       
 +   └─ pulumi:providers:random                                         random                          created (0.00s)     

 + 4 created
```

## Prerequisites

- **AWS EKS Cluster:** An existing EKS cluster stack reference (e.g. `rasa/eks-cluster/ci`) exporting `vpc_id` and `private_subnets`.
- **Pulumi CLI:** Installed and logged in.
- **Python 3.8+** with `pulumi` and `pulumi_aws` packages.
- **AWS CLI & Credentials:** Permissions for RDS, VPC, and Security Groups.
- **Pulumi config:** Set a fixed master password (see Deployment).

## Configuration

The following variables are configured in the script:

| Variable               | Description                                  | Value                          |
|------------------------|----------------------------------------------|--------------------------------|
| `project_name`         | Prefix for all resources                     | `rasa-pro`                     |
| `db_name`              | Database name                                | `integrationtestsdb`           |
| `db_instance_class`    | RDS instance class                           | `db.t4g.small`                 |
| `db_allocated_storage` | Storage size in GB                           | `50`                           |
| `db_storage_type`      | Storage type                                 | `gp3`                          |
| `db_engine`            | Database engine                              | `postgres`                     |
| `db_engine_version`    | PostgreSQL version                           | `16.6`                         |

## Recreate RDS with PostgreSQL 16.x

RDS does not support in-place engine version changes. To switch from 17.x to 16.6 (e.g. for IAM auth):

1. **Set engine version** in `__main__.py`:
   ```python
   db_engine_version = "16.6"
   ```
   (Already set if you are following this guide.)

2. **Destroy the current RDS stack** (from `infrastructure/eks-integration-tests/`):
   ```bash
   pulumi destroy --yes
   ```
   This removes the RDS instance, subnet group, and security group. The new instance will use the same fixed master password from config.

3. **Create the new RDS instance**:
   ```bash
   pulumi up --yes
   ```
   Wait for the RDS instance to reach "available" (several minutes).

4. **Run the database setup** from inside the VPC (jumphost or one-off Job):
   - Create `integration_tests_user_no_password` and grant `rds_iam`.
   - Create the assistant user/database if you use them.
   ```bash
   # From jumphost (or cluster Job) with PGHOST, PGPORT, PGPASSWORD, etc. set:
   ./scripts/setup-databases.sh
   ```
   Or run the bootstrap SQL manually (see "Run bootstrap from inside the cluster" in this README).

5. **Update eks-helm** (if the RDS endpoint changed):  
   Pulumi will create a new RDS instance with a new endpoint. If your eks-helm stack reads the DB endpoint from this stack (stack reference), run `pulumi up` in eks-helm so it picks up the new endpoint. If the endpoint is hardcoded, update it and redeploy.

6. **Restart Rasa** in the cluster so it reconnects to the new DB (e.g. rollout restart of the Rasa deployment).

## Deployment

1. **Select or create** your Pulumi stack:
   ```bash
   pulumi stack init <org>/rasa-pro-integration-tests/dev
   pulumi stack select dev
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set the fixed master password** (required; stored as a secret):
   ```bash
   pulumi config set --secret db_password "your-secure-password"
   ```
   Use a strong password; RDS accepts printable ASCII except `/`, `@`, `"`, and space.

4. **Preview and apply**:
   ```bash
   pulumi preview
   pulumi up --yes
   ```

## Outputs

Note: `db_password` is the value you set in config; it is exported for use by scripts (e.g. `PGPASSWORD=$(pulumi stack output db_password)`).

| Output Name           | Description                              |
|-----------------------|------------------------------------------|
| `db_host`             | RDS endpoint hostname                    |
| `db_port`             | Database port (5432)                     |
| `db_name`             | Database name                            |
| `db_username`         | Database username                        |
| `db_password`         | Database master password (from config secret; set before first deploy) |
| `db_connection_string`| Complete PostgreSQL connection string   |

## Database Connection

### From within the EKS cluster:

You can connect to the database using the connection string or individual components:

```bash
# Using the connection string
psql "postgresql://integration_tests_user:<password>@<db_host>:5432/integrationtestsdb"

# Or using individual components
psql -h <db_host> -p 5432 -U integration_tests_user -d integrationtestsdb
```

### From Pulumi Secrets:

Retrieve credentials from Pulumi secrets:

```bash
pulumi stack output db_password
pulumi stack output db_connection_string
```

## Bootstrap assistant database

After the RDS instance is created, run the setup script to create the assistant database and user (including `rds_iam` for the assistant user).

**Required environment variables:**

- **Connection:** `PGHOST`, `PGUSER`, `PGPASSWORD` (optional: `PGPORT`, `PGDATABASE`)
- **Assistant:** `DB_ASSISTANT_USERNAME`, `DB_ASSISTANT_DATABASE`

**Example (from Pulumi outputs):**

From the `eks-integration-tests` directory, select your stack and set env vars, then run the script:

```bash
cd infrastructure/eks-integration-tests
pulumi stack select rasa/integration-tests/ci   # or your stack name

# Connection (from Pulumi stack outputs)
export PGHOST=$(pulumi stack output db_host)
export PGPORT=$(pulumi stack output db_port)
export PGUSER=$(pulumi stack output db_username)
export PGPASSWORD=$(pulumi stack output db_password)
export PGDATABASE=postgres

# Assistant user and database names (must not conflict with existing DB; RDS already has db_name=integrationtestsdb)
export DB_ASSISTANT_USERNAME="${DB_ASSISTANT_USERNAME:-integration_tests_assistant_no_password}"
export DB_ASSISTANT_DATABASE="${DB_ASSISTANT_DATABASE:-assistant_db}"

./scripts/setup-databases.sh
```

The script creates the assistant user and database and also creates `integration_tests_user_no_password` with `rds_iam` (for RDS IAM auth from eks-helm). Use a `DB_ASSISTANT_DATABASE` name that does not already exist (the RDS instance already has `integrationtestsdb`).

The script enforces SSL by setting `PGSSLMODE=require` and creates the assistant user, grants `rds_iam`, and creates the assistant database with the correct ownership.

### SSM into jumphost and run setup-databases.sh

If you use the **jumphost** (infrastructure/jumphost) to reach RDS:

1. **Ensure jumphost is deployed** and RDS allows it:
   - Deploy jumphost: `cd infrastructure/jumphost && pulumi stack select rasa/jumphost/dev && pulumi up --yes` (or your jumphost stack).
   - RDS security group must allow port 5432 from the **jumphost security group** (add an ingress rule: source = jumphost SG). Get jumphost SG: `pulumi stack output jumphost_security_group_id` from the jumphost stack.

2. **Get the SSM connect command** (from your machine, with AWS credentials and Session Manager plugin installed):
   ```bash
   cd infrastructure/jumphost
   pulumi stack select rasa/jumphost/dev   # or your jumphost stack
   pulumi stack output connect_via_ssm
   ```
   Copy the printed command (e.g. `aws ssm start-session --target i-xxxxx --region eu-west-1`).

3. **Start an SSM session** to the jumphost:
   ```bash
   aws ssm start-session --target <instance-id> --region eu-west-1
   ```
   (Use the instance ID from the command in step 2.) You will be logged in as `ssm-user` (Amazon Linux 2023). Switch to `ec2-user` if the jumphost scripts are there: `sudo su - ec2-user`.

4. **On the jumphost**, the jumphost user_data already set up `connect-db.sh` and `.pgpass` with RDS connection (from the integration-tests/ci stack ref). To run the full setup script you need the script and env:
     ```
   - **Run the bootstrap SQL manually** As `ec2-user`, use the existing connection:
     ```bash
     source /home/ec2-user/connect-db.sh
     export PGSSLMODE=require
     export PGPASSWORD=<master password available with `pulumi stack output db_password --show-secrets`>
     psql -c "CREATE USER integration_tests_user_no_password; GRANT rds_iam TO integration_tests_user_no_password;"
     ```
     That creates the IAM user for the tracker store. For the assistant user and database, run the equivalent `CREATE USER`, `GRANT rds_iam`, `CREATE DATABASE`, etc. from `setup-databases.sh`.

5. **Exit the session**: type `exit`.

**Prerequisites:** AWS CLI and [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) installed locally; IAM permissions for `ssm:StartSession` on the jumphost instance.

## Security

- **Network Isolation:** Database is only accessible from within the VPC (10.0.0.0/16)
- **No Public Access:** `publicly_accessible` is set to `false`
- **Encrypted Storage:** RDS uses AWS managed encryption
- **Secrets Management:** Credentials are managed through Pulumi's built-in secrets system

## Cleanup
NOTE : The DB is a one off setup that is used by the helm-workflow to run tests so its not recommended to clean up the DB. If however you need to update it then the below steps can be used to clean up the old DB.

To destroy resources:
```bash
pulumi destroy --yes
```

To remove the stack:
```bash
pulumi stack rm dev --yes
```

## Notes

### VPC Configuration

The database uses the existing EKS cluster VPC and private subnets:
- **VPC ID:** Retrieved from the stack reference (e.g. `rasa/eks-cluster/ci`)
- **Private Subnets:** Used for DB subnet group
- **Security Group:** Allows PostgreSQL access only from VPC CIDR

### Database Configuration

- **Backup:** 7-day retention with daily backup window
- **Maintenance:** Weekly maintenance window
- **Deletion Protection:** Disabled for integration tests
- **Final Snapshot:** Skipped for integration tests

## Troubleshooting

### Connection Issues

1. **Check Security Group:** Ensure the security group allows port 5432 from your source
2. **Verify VPC:** Confirm you're connecting from within the VPC
3. **Check RDS Status:** Ensure the RDS instance is in "available" state

### PAM authentication failed (RDS IAM user)

If the tracker store fails with **"PAM authentication failed for user integration_tests_user_no_password"** even though the user exists and is in `rds_iam` (e.g. `\du` shows `integration_tests_user_no_password | {} | {rds_iam}`), 
RDS **PostgreSQL 17.x** has a known issue: `pg_hba.conf` can show `pam` instead of `iam` for `+rds_iam` connections, so IAM auth is rejected. See [AWS repost](https://repost.aws/questions/QUZ_tdsuqQTQO9hRXN14uGsg/rds-postgresql-17-5-iam-auth-fails-pg-hba-conf-shows-pam-instead-of-iam-for-rds-iam).

**Workarounds:**

1. **Use PostgreSQL 16 for new RDS** — In `__main__.py`, set `db_engine_version = "16.6"` (or another 16.x available in your region, e.g. 16.11 in eu-west-1) and recreate the RDS instance so IAM auth works.
2. **Use the password user for the tracker store** — In eks-helm, configure the tracker store to use `integration_tests_user` (master) and set the password from the RDS master secret (e.g. `trackerStoreDbPassword` from Pulumi/GitHub secret). That avoids IAM auth until AWS fixes PostgreSQL 17.

### Common Commands

```bash
# Check RDS instance status
aws rds describe-db-instances --db-instance-identifier rasa-pro-integration-tests-db

# Check security group rules
aws ec2 describe-security-groups --group-names rasa-pro-integration-tests-db-sg

# Get database credentials
pulumi stack output db_password
pulumi stack output db_connection_string
```
