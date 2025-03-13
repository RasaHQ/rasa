import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s
import json
import os

# Variables
project_name = "rasa-pro"
aws_region = os.environ.get("AWS_REGION", "eu-west-1")
account_id = aws.get_caller_identity().account_id

# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/rasa-pro-eks-base/rasa-pro-eks-base")
eks_cluster_name = stack_ref.get_output("eks_cluster_name")
vpc_id = stack_ref.get_output("vpc_id")

# 📌 **Step 1: Get EKS Cluster and Generate Kubeconfig**
eks_cluster = aws.eks.get_cluster_output(name=eks_cluster_name)

def create_kubeconfig(cluster):
    cert_auth = getattr(cluster, "certificate_authority", {})
    cert_data = cert_auth.get("data", "") if isinstance(cert_auth, dict) else ""
    cluster_info = {"server": cluster.endpoint}
    if cert_data:
        cluster_info["certificate-authority-data"] = cert_data
    else:
        cluster_info["insecure-skip-tls-verify"] = True
    return json.dumps({
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{
            "name": cluster.name,
            "cluster": cluster_info
        }],
        "contexts": [{
            "name": cluster.name,
            "context": {
                "cluster": cluster.name,
                "user": "aws"
            }
        }],
        "current-context": cluster.name,
        "users": [{
            "name": "aws",
            "user": {
                "exec": {
                    "apiVersion": "client.authentication.k8s.io/v1beta1",
                    "command": "aws",
                    "args": ["eks", "get-token", "--cluster-name", cluster.name]
                }
            }
        }]
    })

kubeconfig = eks_cluster.apply(create_kubeconfig)
k8s_provider = k8s.Provider(f"{project_name}-k8s-provider", kubeconfig=kubeconfig)

# 📌 **Step 2: Create OIDC Provider for the EKS Cluster**
# Use aws.eks.get_cluster to retrieve full cluster info including identities
# Extract the OIDC issuer URL using the identities field
cluster_info = aws.eks.get_cluster(name=eks_cluster_name)
oidc_url = pulumi.Output.from_input(cluster_info).apply(
    lambda info: info.identities[0]["oidcs"][0]["issuer"]
)

# Create the OIDC provider with the required client IDs and thumbprint
# The thumbprint below is the commonly used thumbprint for Amazon's CA
oidc_provider = aws.iam.OpenIdConnectProvider(
    f"{project_name}-oidc-provider",
    client_id_lists=["sts.amazonaws.com"],
    thumbprint_lists=["9e99a48a9960b14926bb7f3b02e22da0b1a65b1c"],
    url=oidc_url
)

# 📌 **Step 3: Load IAM Policy JSON for LB Controller from External File**
with open("json/lb-controller-policy.json") as f:
    lb_controller_policy_json = json.load(f)

lb_controller_policy = aws.iam.Policy(
    f"{project_name}-lb-controller-policy",
    description="Policy for AWS Load Balancer Controller",
    policy=json.dumps(lb_controller_policy_json)
)

# 📌 **Step 4: Create IAM Role Trust Policy for the LB Controller**
def build_assume_role_policy(oidc_url, oidc_provider_arn):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Federated": oidc_provider_arn
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        f"{oidc_url}:sub": "system:serviceaccount:kube-system:aws-load-balancer-controller"
                    }
                }
            }
        ]
    }
    return json.dumps(policy)

assume_role_policy = pulumi.Output.all(oidc_provider.url, oidc_provider.arn).apply(
    lambda args: build_assume_role_policy(args[0], args[1])
)

lb_controller_role = aws.iam.Role(
    f"{project_name}-lb-controller-role",
    assume_role_policy=assume_role_policy
)

# 📌 **Step 5: Attach the IAM Policy to the Role**
aws.iam.RolePolicyAttachment(
    f"{project_name}-lb-controller-role-policy-attach",
    role=lb_controller_role.name,
    policy_arn=lb_controller_policy.arn
)

# 📌 **Step 6: Deploy the AWS Load Balancer Controller using Helm**
lb_controller_helm = k8s.helm.v3.Chart(
    f"{project_name}-aws-load-balancer-controller",
    k8s.helm.v3.ChartOpts(
        chart="aws-load-balancer-controller",
        version="1.11.0",
        fetch_opts=k8s.helm.v3.FetchOpts(
            repo="https://aws.github.io/eks-charts"
        ),
        namespace="kube-system",
        values={
            "clusterName": eks_cluster_name,
            "serviceAccount": {
                "create": True,
                "name": "aws-load-balancer-controller",
                "annotations": {
                    "eks.amazonaws.com/role-arn": lb_controller_role.arn
                }
            },
            "region": aws_region,
            "vpcId": vpc_id
        }
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# 📌 **Step 7: Export Outputs for Verification**
pulumi.export("aws_lb_controller_iam_role", lb_controller_role.arn)
pulumi.export("aws_lb_controller_chart", f"{project_name}-aws-load-balancer-controller")
