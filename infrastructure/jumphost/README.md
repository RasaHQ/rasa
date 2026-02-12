## 3. jumphost

The **jumphost** stack provisions a small EC2 bastion instance in the same VPC as the EKS cluster. It is used to access and debug the **RDS database** (PostgreSQL) created by **eks-integration-tests** without exposing the database to the public internet.

**What it creates:**

- **EC2 instance**: Amazon Linux 2023 (`t3.micro`) in a public subnet, with an IAM role that allows **AWS Systems Manager (SSM)** Session Manager access (no SSH keys required).
- **Security group**: Allows SSH (and can be tightened to specific IPs). You must update the RDS security group to allow inbound traffic from this jumphost security group so the instance can reach the database.

*** IMPORTANT ***: Allow the jumphost to reach RDS (PostgreSQL). Do one of:

   Add the rule manually:
   1. Get the jumphost security group ID:
      pulumi stack output jumphost_security_group_id
   2. Get the RDS security group ID from the integration-tests stack:
      pulumi stack output -s rasa/eks-integration-tests/ci db_security_group_id
   3. Add an inbound rule to the RDS security group:
      - In AWS Console: EC2 → Security Groups → select RDS SG → Edit inbound rules
        → Add rule: Type = PostgreSQL, Port = 5432, Source = jumphost SG (paste ID from step 1)
      - Or via AWS CLI:
        aws ec2 authorize-security-group-ingress \\
          --group-id <RDS_SG_ID> \\
          --protocol tcp --port 5432 \\
          --source-group <JUMPHOST_SG_ID> \\
          --region eu-west-1

- **Stack reference**: Requires `stackRefName` config pointing to the **eks-integration-tests** stack; it reads `db_host`, `db_port`, `db_name`, `db_username`, and `db_password` from that stack to configure the jumphost.

**On the instance:**

- PostgreSQL 15 client and tools (`psql`, `jq`, `vim`, `htop`).
- Helper scripts for `ec2-user`:
  - `./connect-db.sh` — interactive `psql` session to the RDS database (SSL enabled).
  - `./check-db-roles.sh` — lists roles, permissions, and database info for debugging access issues.
- Connection details are stored in `~/.pgpass` for passwordless `psql` use.

**How to connect:**

1. **Via SSM (recommended):** `pulumi stack output connect_via_ssm` then run the printed command (e.g. `aws ssm start-session --target <instance-id> --region eu-west-1`).
2. **Via SSH:** `pulumi stack output connect_via_ssh` (requires your SSH key to be available to the instance).

After connecting, run `cat README.txt` on the jumphost for a short usage guide.

**Destroy jumphost after using:**

1. From the jumphost project directory, select the stack:
   ```bash
   cd infrastructure/jumphost
   pulumi stack select rasa/jumphost/<stack>   # e.g. dev or ci
   ```
2. Destroy all resources (EC2 instance, security group, IAM role/profile, and the RDS SG ingress rule):
   ```bash
   pulumi destroy --yes
   ```
   Or run `pulumi destroy` without `--yes` to review and confirm.
3. (Optional) Remove the stack from Pulumi:
   ```bash
   pulumi stack rm rasa/jumphost/<stack> --force
   ```
