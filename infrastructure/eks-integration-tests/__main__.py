import pulumi
import pulumi_aws as aws

# Variables
project_name = "rasa-pro"
db_name = "integrationtestsdb"
db_instance_class = "db.t4g.small"
db_allocated_storage = 50
db_storage_type = "gp3"
db_engine = "postgres"
# RDS PostgreSQL 17.x has a known pg_hba bug: IAM auth can fail with "PAM authentication failed".
# Use a 16.x version available in your region (eu-west-1: 16.6, 16.8, 16.9, 16.10, 16.11).
db_engine_version = "16.6"

# Fixed master password from Pulumi config (stored as secret). Set with:
#   pulumi config set --secret db_password "your-secure-password"
# RDS accepts printable ASCII except /, @, " and space.
config = pulumi.Config()
db_password = config.get_secret("db_password")
# use pulumi stack output db_password --show-secrets to get the password
if not db_password:
    raise ValueError(
        "config 'integration-tests:db_password' is required. "
        "Set it with: pulumi config set --secret db_password \"your-secure-password\""
    )

# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/eks-cluster/ci")
# vpc_id = stack_ref.get_output("vpc_id")
# Hardcoded VPC ID (from eks-cluster/ci stack)
vpc_id = "vpc-05c25d975207885ea"
# Hardcoded private subnet IDs (from eks-cluster/ci stack)
# private_subnets = stack_ref.get_output("private_subnets")
private_subnets = [
    "subnet-0a47071e225de6910",
    "subnet-0a5f6f4c061428b59",
]

# 📌 Step 1: Create DB Subnet Group
db_subnet_group = aws.rds.SubnetGroup(
    f"{project_name}-integration-tests-db-subnet-group",
    name=f"{project_name}-integration-tests-db-subnet-group",
    subnet_ids=private_subnets,
    tags={
        "Name": f"{project_name}-integration-tests-db-subnet-group",
        "Project": project_name,
        "Environment": "integration-tests"
    }
)

# 📌 Step 2: Create Security Group for RDS
db_security_group = aws.ec2.SecurityGroup(
    f"{project_name}-integration-tests-db-sg",
    name=f"{project_name}-integration-tests-db-sg",
    description="Security group for integration tests RDS database",
    vpc_id=vpc_id,
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            from_port=5432,
            to_port=5432,
            protocol="tcp",
            cidr_blocks=["10.0.0.0/16"],  # Allow access from VPC CIDR
            description="PostgreSQL access from VPC"
        )
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            from_port=0,
            to_port=0,
            protocol="-1",
            cidr_blocks=["0.0.0.0/0"],
            description="All outbound traffic"
        )
    ],
    tags={
        "Name": f"{project_name}-integration-tests-db-sg",
        "Project": project_name,
        "Environment": "integration-tests"
    }
)

# 📌 Step 3: Create RDS Instance
db_instance = aws.rds.Instance(
    f"{project_name}-integration-tests-db",
    identifier=f"{project_name}-integration-tests-db",
    db_name=db_name,
    instance_class=db_instance_class,
    allocated_storage=db_allocated_storage,
    storage_type=db_storage_type,
    engine=db_engine,
    engine_version=db_engine_version,
    username="integration_tests_user",
    password=db_password,
    vpc_security_group_ids=[db_security_group.id],
    db_subnet_group_name=db_subnet_group.name,
    backup_retention_period=7,
    backup_window="03:00-04:00",
    maintenance_window="sun:04:00-sun:05:00",
    multi_az=False, 
    publicly_accessible=False,  # Only accessible within VPC
    iam_database_authentication_enabled=True,  # Required for rds_iam role and IAM-auth users (e.g. integration_tests_user_no_password)
    skip_final_snapshot=True,  
    deletion_protection=False,
    tags={
        "Name": f"{project_name}-integration-tests-db",
        "Project": project_name,
        "Environment": "integration-tests"
    }
)

# 📌 Step 4: Export outputs
pulumi.export("db_host", db_instance.endpoint)
pulumi.export("db_port", db_instance.port)
pulumi.export("db_name", db_instance.db_name)
pulumi.export("db_username", db_instance.username)
pulumi.export("db_password", db_password)
pulumi.export("db_connection_string", pulumi.Output.all(
    db_instance.endpoint,
    db_instance.port,
    db_instance.db_name,
    db_instance.username,
).apply(lambda args: f"postgresql://{args[3]}:{db_password}@{args[0]}:{args[1]}/{args[2]}"))

