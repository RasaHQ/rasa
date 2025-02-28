import pulumi
import pulumi_aws as aws
import pulumi_awsx as awsx
import json

# Variables
project_name = "rasa-pro"
aws_region = "eu-west-1"
account_id = aws.get_caller_identity().account_id

# 📌 **Step 1: Create VPC using awsx.ec2.Vpc**
vpc = awsx.ec2.Vpc(
    f"{project_name}-vpc",
    cidr_block="10.0.0.0/16",
    number_of_availability_zones=2,
    subnet_strategy="Auto",
    subnet_specs=[
        awsx.ec2.SubnetSpecArgs(
            type=awsx.ec2.SubnetType.PUBLIC,
            cidr_mask=24,
        ),
        awsx.ec2.SubnetSpecArgs(
            type=awsx.ec2.SubnetType.PRIVATE,
            cidr_mask=24,
        ),
    ],
    nat_gateways=awsx.ec2.NatGatewayConfigurationArgs(
        strategy=awsx.ec2.NatGatewayStrategy.SINGLE
    ),
    tags={"Name": f"{project_name}-vpc"},
)

# 📌 **Step 2: Extract Subnets and Gateway**
public_subnets = vpc.public_subnet_ids
private_subnets = vpc.private_subnet_ids
internet_gateway = vpc.internet_gateway

# 📌 **Step 3: IAM Roles and EKS Cluster**
eks_cluster_role = aws.iam.Role(
    f"{project_name}-eks-cluster-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "eks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    })
)

aws.iam.RolePolicyAttachment(
    f"{project_name}-eks-cluster-policy-1",
    role=eks_cluster_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
)

aws.iam.RolePolicyAttachment(
    f"{project_name}-eks-cluster-policy-2",
    role=eks_cluster_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
)

eks_cluster = aws.eks.Cluster(
    f"{project_name}-cluster",
    name = project_name + "-cluster",
    role_arn=eks_cluster_role.arn,
    vpc_config=aws.eks.ClusterVpcConfigArgs(
        subnet_ids=pulumi.Output.all(public_subnets, private_subnets)
            .apply(lambda subnet_lists: subnet_lists[0] + subnet_lists[1]),
        endpoint_private_access=True,
        endpoint_public_access=True,
    ),
)

# 📌 **Step 4: Create IAM Role for Fargate Pods with Trust Policy**
fargate_pod_execution_role = aws.iam.Role(
    f"{project_name}-fargate-pod-execution-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "eks-fargate-pods.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Effect": "Allow",
                "Action": "sts:AssumeRole",
                "Principal": {
                    "Service": "eks-fargate-pods.amazonaws.com"
                },
                "Condition": {
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:eks:{aws_region}:{account_id}:fargateprofile/{project_name}-cluster/{project_name}-fargate-profile/*"
                    }
                }
            }
        ]
    })
)

aws.iam.RolePolicyAttachment(
    f"{project_name}-fargate-policy-1",
    role=fargate_pod_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEKSFargatePodExecutionRolePolicy"
)

aws.iam.RolePolicyAttachment(
    f"{project_name}-fargate-ecr-policy",
    role=fargate_pod_execution_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
)

# 📌 **Step 5: Create the Fargate Profile**
fargate_profile = aws.eks.FargateProfile(
    f"{project_name}-fargate-profile",
    fargate_profile_name = project_name + "-fargate-profile",
    cluster_name = eks_cluster.name,
    pod_execution_role_arn=fargate_pod_execution_role.arn,
    subnet_ids=private_subnets,
    selectors=[
        aws.eks.FargateProfileSelectorArgs(namespace="default"),
        aws.eks.FargateProfileSelectorArgs(namespace="kubernetes-dashboard"),
        aws.eks.FargateProfileSelectorArgs(namespace="kube-system"),
        aws.eks.FargateProfileSelectorArgs(namespace="rasa-pro-*"),
    ],
)

# 📌 **Step 6: Export Outputs**
pulumi.export("vpc_id", vpc.vpc_id)
pulumi.export("public_subnets", public_subnets)
pulumi.export("private_subnets", private_subnets)
pulumi.export("internet_gateway", internet_gateway.id)
pulumi.export("eks_cluster_name", eks_cluster.name)
pulumi.export("eks_cluster_role", eks_cluster_role.id)