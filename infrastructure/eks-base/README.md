# Rasa Pro Pulumi Base Infrastructure

## Overview
This Pulumi script written in Python provisions AWS base infrastructure for running Rasa Pro cluster using AWS Fargate. The setup includes:

- A Virtual Private Cloud (VPC)
- Public and private subnets
- Internet Gateway and NAT Gateway
- Elastic Kubernetes Service (EKS) cluster
- IAM roles and policies for EKS and Fargate

## Resources

```sh
     Type                                          Name                                 Status              Info
 +   pulumi:pulumi:Stack                           rasa-pro-eks-base-rasa-pro-eks-base  created (763s)      
 +   ├─ aws:iam:Role                               rasa-pro-eks-cluster-role            created (1s)
 +   ├─ aws:iam:Role                               rasa-pro-fargate-pod-execution-role  created (1s)
 +   ├─ awsx:ec2:Vpc                               rasa-pro-vpc                         created (3s)
 +   │  └─ aws:ec2:Vpc                             rasa-pro-vpc                         created (3s)
 +   │     ├─ aws:ec2:Subnet                       rasa-pro-vpc-private-1               created (1s)
 +   │     │  └─ aws:ec2:RouteTable                rasa-pro-vpc-private-1               created (1s)
 +   │     │     ├─ aws:ec2:RouteTableAssociation  rasa-pro-vpc-private-1               created (1s)
 +   │     │     └─ aws:ec2:Route                  rasa-pro-vpc-private-1               created (1s)
 +   │     ├─ aws:ec2:Subnet                       rasa-pro-vpc-private-2               created (2s)
 +   │     │  └─ aws:ec2:RouteTable                rasa-pro-vpc-private-2               created (1s)
 +   │     │     ├─ aws:ec2:RouteTableAssociation  rasa-pro-vpc-private-2               created (0.66s)
 +   │     │     └─ aws:ec2:Route                  rasa-pro-vpc-private-2               created (1s)
 +   │     ├─ aws:ec2:InternetGateway              rasa-pro-vpc                         created (1s)
 +   │     ├─ aws:ec2:Subnet                       rasa-pro-vpc-public-2                created (11s)
 +   │     │  └─ aws:ec2:RouteTable                rasa-pro-vpc-public-2                created (1s)
 +   │     │     ├─ aws:ec2:Route                  rasa-pro-vpc-public-2                created (2s)
 +   │     │     └─ aws:ec2:RouteTableAssociation  rasa-pro-vpc-public-2                created (2s)
 +   │     └─ aws:ec2:Subnet                       rasa-pro-vpc-public-1                created (12s)
 +   │        ├─ aws:ec2:Eip                       rasa-pro-vpc-1                       created (1s)
 +   │        ├─ aws:ec2:RouteTable                rasa-pro-vpc-public-1                created (2s)
 +   │        │  ├─ aws:ec2:RouteTableAssociation  rasa-pro-vpc-public-1                created (1s)
 +   │        │  └─ aws:ec2:Route                  rasa-pro-vpc-public-1                created (2s)
 +   │        └─ aws:ec2:NatGateway                rasa-pro-vpc-1                       created (95s)
 +   ├─ aws:iam:RolePolicyAttachment               rasa-pro-eks-cluster-policy-2        created (0.63s)
 +   ├─ aws:iam:RolePolicyAttachment               rasa-pro-eks-cluster-policy-1        created (1s)
 +   ├─ aws:iam:RolePolicyAttachment               rasa-pro-fargate-ecr-policy          created (1s)
 +   ├─ aws:iam:RolePolicyAttachment               rasa-pro-fargate-policy-1            created (2s)
 +   ├─ aws:eks:Cluster                            rasa-pro-cluster                     created (465s)
 +   └─ aws:eks:FargateProfile                     rasa-pro-fargate-profile             created (168s)
 + 30 created
```

## Prerequisites

Ensure you have the following installed before running the Pulumi script:

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- [AWS CLI](https://aws.amazon.com/cli/) (configured with the appropriate credentials)
- Python (if using a virtual environment, activate it)

## Infrastructure components

### 1. **VPC and Subnets**
- Creates a VPC (`10.0.0.0/16`)
- Public subnets (`10.0.1.0/24`, `10.0.2.0/24`)
- Private subnets (`10.0.3.0/24`, `10.0.4.0/24`)
- Associates subnets with appropriate route tables

### 2. **Networking Components**
- Internet Gateway for public subnets
- NAT Gateway for private subnets
- Route tables for public and private traffic handling

### 3. **IAM Roles and Policies**
- IAM role for EKS cluster with required policies
- IAM role for Fargate pod execution

### 4. **EKS Cluster & Fargate Profile**
- EKS cluster with public and private subnet access
- Fargate profile for running Rasa workloads

## Deployment

1. **Clone the repository** and navigate into the directory:

   ```sh
   git clone <this-repo-url>
   cd <this-repo-directory>/infrastructure/eks-base
   ```

2. **Login to Pulumi** (if not already logged in):

   ```sh
   pulumi login
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Initialize the Pulumi stack**:

   ```sh
   pulumi stack init rasa-pro-eks-base
   ```

5. **Set AWS Region (optional)**:

   ```sh
   pulumi config set aws:region eu-west-1
   ```

6. **Deploy the infrastructure**:

   ```sh
   pulumi up
   ```

   This will show the changes and prompt for confirmation before deploying.

7. Outputs

| Output Name          | Description                   |
|----------------------|-------------------------------|
| `vpc_id`            | ID of the created VPC         |
| `public_subnets`    | List of public subnet IDs     |
| `private_subnets`   | List of private subnet IDs    |
| `internet_gateway`  | ID of the Internet Gateway    |
| `eks_cluster_name`  | Name of the EKS cluster       |
| `eks_cluster_role`  | IAM Role for EKS cluster      |


## Cleanup

To destroy all resources and remove the infrastructure:

```sh
pulumi destroy
```

To remove the Pulumi stack completely:

```sh
pulumi stack rm dev  # Replace 'dev' with your stack name
```

## Notes

- The cluster **uses AWS Fargate**, eliminating the need for EC2 instances.
- No manual node group scaling is required.
- IAM roles are created for both the **EKS cluster** and the **Fargate profile**.
- Kubernetes workloads run on Fargate without requiring dedicated EC2 instances.

## Contributions

Contributions are welcome! Please submit a pull request or open an issue for discussion.
