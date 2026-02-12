import pulumi
import pulumi_aws as aws

config = pulumi.Config()
project_name = config.get("projectName") or "rasa-pro"
stack_ref_name = config.require("stackRefName")

# Hardcoded VPC and subnet values (from your EKS cluster)
vpc_id = "vpc-05c25d975207885ea"

# Public subnets for jumphost
public_subnets = [
    "subnet-01ccdcf5f2b0577d8",  # rasa-pro-ci-cluster-public-1 (eu-west-1a)
    "subnet-00eb6d0228bc97f3b",  # rasa-pro-ci-cluster-public-2 (eu-west-1b)
]

# Get RDS info from the stack reference
stack_ref = pulumi.StackReference(stack_ref_name)
db_endpoint = stack_ref.get_output("db_host")
db_port = stack_ref.get_output("db_port")
db_name = stack_ref.get_output("db_name")
db_username = stack_ref.get_output("db_username")
db_password = stack_ref.get_output("db_password")

# Create security group for jumphost
jumphost_sg = aws.ec2.SecurityGroup("jumphostSecurityGroup",
    name=f"{project_name}-jumphost-sg",
    description="Security group for jumphost",
    vpc_id=vpc_id,  # Now using the hardcoded VPC ID
    ingress=[
        # SSH access - you can restrict this to your IP for better security
        aws.ec2.SecurityGroupIngressArgs(
            from_port=22,
            to_port=22,
            protocol="tcp",
            cidr_blocks=["0.0.0.0/0"],  # TODO: Change to your IP
            description="SSH access"
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            from_port=0,
            to_port=0,
            protocol="-1",
            cidr_blocks=["0.0.0.0/0"],
            description="All outbound traffic"
        ),
    ],
    tags={
        "Name": f"{project_name}-jumphost-sg",
        "Project": project_name,
    }
)

# Get the latest Amazon Linux 2023 AMI
ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["amazon"],
    filters=[
        aws.ec2.GetAmiFilterArgs(
            name="name",
            values=["al2023-ami-*-x86_64"]
        ),
        aws.ec2.GetAmiFilterArgs(
            name="virtualization-type",
            values=["hvm"]
        ),
    ]
)

# Create IAM role for the EC2 instance (for SSM access)
jumphost_role = aws.iam.Role("jumphostRole",
    name=f"{project_name}-jumphost-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }""",
    tags={
        "Name": f"{project_name}-jumphost-role",
        "Project": project_name,
    }
)

# Attach SSM policy for Session Manager access
ssm_policy_attachment = aws.iam.RolePolicyAttachment("jumphostSsmPolicy",
    role=jumphost_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
)

# Create instance profile
jumphost_profile = aws.iam.InstanceProfile("jumphostProfile",
    name=f"{project_name}-jumphost-profile",
    role=jumphost_role.name
)

# User data script to install PostgreSQL client and tools
def _build_user_data(args):
    endpoint_raw = args[0]
    host = str(endpoint_raw).strip().split(":")[0] if endpoint_raw else ""
    port = int(float(args[1])) if args[1] is not None else 5432
    db = args[2] or "postgres"
    user = args[3] or ""
    # Escape single quotes in password for shell
    pw = (args[4] or "").replace("'", "'\"'\"'")
    return f"""#!/bin/bash
set -e

# Update system
dnf update -y

# Install PostgreSQL 15 client
dnf install -y postgresql15

# Install useful tools
dnf install -y jq vim htop

# Create connection script (values embedded so it works when run by ec2-user)
# RDS requires SSL; PGSSLMODE=require avoids "no pg_hba.conf entry ... no encryption"
cat > /home/ec2-user/connect-db.sh << 'SCRIPT'
#!/bin/bash
export PGHOST="{host}"
export PGPORT={port}
export PGDATABASE="{db}"
export PGUSER="{user}"
export PGPASSWORD='{pw}'
export PGSSLMODE=require

echo "Connecting to PostgreSQL..."
echo "Host: $PGHOST"
echo "Database: $PGDATABASE"
echo "User: $PGUSER"
psql
SCRIPT

chmod +x /home/ec2-user/connect-db.sh
chown ec2-user:ec2-user /home/ec2-user/connect-db.sh

# Create DB roles check script
cat > /home/ec2-user/check-db-roles.sh << 'SCRIPT'
#!/bin/bash
export PGHOST="{host}"
export PGPORT={port}
export PGDATABASE="{db}"
export PGUSER="{user}"
export PGPASSWORD='{pw}'
export PGSSLMODE=require

echo "=== Database Roles and Users ==="
psql -c "\\\\du"

echo ""
echo "=== Database List ==="
psql -c "\\\\l"

echo ""
echo "=== Current Database Permissions ==="
psql -c "SELECT datname, datacl FROM pg_database WHERE datname = '$PGDATABASE';"

echo ""
echo "=== Schema Permissions ==="
psql -c "SELECT schema_name, schema_owner FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema');"

echo ""
echo "=== Table Permissions ==="
psql -c "SELECT grantee, privilege_type, table_schema, table_name FROM information_schema.table_privileges WHERE table_schema = 'public' LIMIT 20;"

echo ""
echo "=== All Roles (including RDS IAM) ==="
psql -c "SELECT rolname, rolcanlogin, rolsuper, rolinherit, rolcreaterole, rolcreatedb FROM pg_roles ORDER BY rolname;"

echo ""
echo "=== Check for rds_iam role ==="
psql -c "SELECT rolname FROM pg_roles WHERE rolname = 'rds_iam';"
SCRIPT

chmod +x /home/ec2-user/check-db-roles.sh
chown ec2-user:ec2-user /home/ec2-user/check-db-roles.sh

# Create .pgpass for passwordless access
cat > /home/ec2-user/.pgpass << 'PGPASS'
{host}:{port}:{db}:{user}:{args[4] or ""}
PGPASS

chmod 600 /home/ec2-user/.pgpass
chown ec2-user:ec2-user /home/ec2-user/.pgpass

# Create a README
cat > /home/ec2-user/README.txt << 'README'
=== Jumphost Database Tools ===

Available scripts:
1. ./connect-db.sh       - Connect to PostgreSQL interactively
2. ./check-db-roles.sh   - Display all roles, permissions, and database info

Quick PostgreSQL commands once connected:
- \\du                   - List all roles
- \\l                    - List all databases
- \\dt                   - List tables in current database
- \\c <database>         - Connect to a different database
- \\q                    - Quit psql

Database connection details are stored in ~/.pgpass
README

chown ec2-user:ec2-user /home/ec2-user/README.txt

echo "Jumphost setup complete!"
"""


user_data = pulumi.Output.all(db_endpoint, db_port, db_name, db_username, db_password).apply(
    _build_user_data
)

# Create the EC2 jumphost instance
jumphost = aws.ec2.Instance("jumphost",
    instance_type="t3.micro",
    ami=ami.id,
    subnet_id=public_subnets[0],  # Use first public subnet
    vpc_security_group_ids=[jumphost_sg.id],
    iam_instance_profile=jumphost_profile.name,
    user_data=user_data,
    associate_public_ip_address=True,
    tags={
        "Name": f"{project_name}-jumphost",
        "Project": project_name,
    }
)

# Export outputs
pulumi.export("jumphost_instance_id", jumphost.id)
pulumi.export("jumphost_public_ip", jumphost.public_ip)
pulumi.export("jumphost_private_ip", jumphost.private_ip)
pulumi.export("jumphost_security_group_id", jumphost_sg.id)
pulumi.export("connect_via_ssm", pulumi.Output.concat(
    "aws ssm start-session --target ", jumphost.id, " --region eu-west-1"
))
pulumi.export("connect_via_ssh", pulumi.Output.concat(
    "ssh ec2-user@", jumphost.public_ip
))
pulumi.export("instructions", """
========================================
Jumphost Deployed Successfully!
========================================

To connect:
1. Via SSM (recommended - no SSH keys needed):
   pulumi stack output connect_via_ssm
   Then run the command shown (no AWS credentials required)

2. Via SSH:
   pulumi stack output connect_via_ssh
   (Requires SSH key - you'll need to add your key to AWS)

Once connected, run:
   - sudo su - ec2-user      (to switch to the ec2-user)
   - ./check-db-roles.sh     (to see all roles and permissions)
   - ./connect-db.sh         (to connect to PostgreSQL)

IMPORTANT: Update your RDS security group to allow access from this jumphost security group!
for more information, see the README.md file.
""")
