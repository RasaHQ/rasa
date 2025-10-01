import pulumi
import pulumi_aws as aws
import pulumi_random as random

# Variables
project_name = "rasa-pro"
db_name = "integrationtestsdb"
db_instance_class = "db.t4g.small"
db_allocated_storage = 50
db_storage_type = "gp3"
db_engine = "postgres"
db_engine_version = "17.4"

# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/rasa-pro-eks-base/ci")
vpc_id = stack_ref.get_output("vpc_id")
private_subnets = stack_ref.get_output("private_subnets")

# Generate random password for the database using Pulumi Random provider
# RDS accepts printable ASCII except /, @, " and space
db_password = random.RandomPassword(
    f"{project_name}-integration-tests-db-password",
    length=16,
    special=True,
    override_special="!#$%&*+-=<>?^_`|~",  # Excludes /, @, " and space
    min_upper=1,
    min_lower=1,
    min_numeric=1,
    min_special=1
)

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
    password=db_password.result,
    vpc_security_group_ids=[db_security_group.id],
    db_subnet_group_name=db_subnet_group.name,
    backup_retention_period=7,
    backup_window="03:00-04:00",
    maintenance_window="sun:04:00-sun:05:00",
    multi_az=False,  # Single-AZ as requested
    publicly_accessible=False,  # Only accessible within VPC
    skip_final_snapshot=True,  # For integration tests, we don't need final snapshot
    deletion_protection=False,  # Allow deletion for integration tests
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
pulumi.export("db_password", db_password.result)
pulumi.export("db_connection_string", pulumi.Output.all(
    db_instance.endpoint,
    db_instance.port,
    db_instance.db_name,
    db_instance.username
).apply(lambda args: f"postgresql://{args[3]}:{db_password.result}@{args[0]}:{args[1]}/{args[2]}"))
