# Rasa Pro AWS Load Balancer Controller Infrastructure

## Overview
This Pulumi script written in Python provisions the AWS Load Balancer Controller into an existing EKS cluster. It leverages AWS IAM and OIDC integration, along with a Helm chart, to configure and deploy the controller in your Kubernetes environment. The setup includes:

- Retrieving the EKS cluster and dynamically generating a kubeconfig
- Extracting the OIDC issuer URL from the cluster's identities
- Creating an OIDC provider for secure IAM role assumptions
- Loading an external IAM policy for the Load Balancer Controller
- Building an IAM role with a trust policy using OIDC
- Deploying the AWS Load Balancer Controller via a Helm chart in a specified VPC
- Exporting key outputs for post-deployment verification

## Prerequisites
Before running this Pulumi script, ensure you have the following installed and configured:
- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- [AWS CLI](https://aws.amazon.com/cli/) with the appropriate credentials
- Python 3.x and the required dependencies (install via `pip install -r requirements.txt`)
- An existing EKS cluster
- The JSON file `json/lb-controller-policy.json` containing the IAM policy for the LB Controller
- Outputs from the base infra to fill in variables values (eks_cluster_name, aws_region, vpc_id)

## Project Structure
```sh
     Type                                                                             Name                                                                           Status
     pulumi:pulumi:Stack                                                              rasa-pro-eks-lb-controller-dev
 +   ├─ aws:iam:Policy                                                                rasa-pro-lb-controller-policy                                                  created (1s)
 +   ├─ pulumi:providers:kubernetes                                                   rasa-pro-k8s-provider                                                          created (0.78s)
 +   ├─ aws:iam:OpenIdConnectProvider                                                 rasa-pro-oidc-provider                                                         created (1s)
 +   ├─ kubernetes:helm.sh/v3:Chart                                                   rasa-pro-aws-load-balancer-controller                                          created (70s)
 +   │  ├─ kubernetes:elbv2.k8s.aws/v1beta1:IngressClassParams                        alb                                                                            created (3s)
 +   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRoleBinding                 rasa-pro-aws-load-balancer-controller-rolebinding                              created (3s)
 +   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:Role                               kube-system/rasa-pro-aws-load-balancer-controller-leader-election-role         created (4s)
 +   │  ├─ kubernetes:networking.k8s.io/v1:IngressClass                               alb                                                                            created (1s)
 +   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRole                        rasa-pro-aws-load-balancer-controller-role                                     created (2s)
 +   │  ├─ kubernetes:admissionregistration.k8s.io/v1:MutatingWebhookConfiguration    aws-load-balancer-webhook                                                      created (3s)
 +   │  ├─ kubernetes:admissionregistration.k8s.io/v1:ValidatingWebhookConfiguration  aws-load-balancer-webhook                                                      created (3s)
 +   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:RoleBinding                        kube-system/rasa-pro-aws-load-balancer-controller-leader-election-rolebinding  created (4s)
 +   │  ├─ kubernetes:apps/v1:Deployment                                              kube-system/rasa-pro-aws-load-balancer-controller                              created (63s)
 +   │  ├─ kubernetes:core/v1:Secret                                                  kube-system/aws-load-balancer-tls                                              created (7s)
 +   │  ├─ kubernetes:apiextensions.k8s.io/v1:CustomResourceDefinition                targetgroupbindings.elbv2.k8s.aws                                              created (1s)
 +   │  ├─ kubernetes:apiextensions.k8s.io/v1:CustomResourceDefinition                ingressclassparams.elbv2.k8s.aws                                               created (0.96s)
 +   │  ├─ kubernetes:core/v1:Service                                                 kube-system/aws-load-balancer-webhook-service                                  created (64s)
 +   │  └─ kubernetes:core/v1:ServiceAccount                                          kube-system/aws-load-balancer-controller                                       created (3s)
 +   ├─ aws:iam:Role                                                                  rasa-pro-lb-controller-role                                                    created (1s)
 +   └─ aws:iam:RolePolicyAttachment                                                  rasa-pro-lb-controller-role-policy-attach                                      created (0.97s)

+ 20 created
```

## Infrastructure components
1. **EKS Cluster Retrieval and Kubeconfig Generation**
   - **EKS Cluster:**  
     The script retrieves details of an existing EKS cluster (`rasa-pro-cluster`) using `aws.eks.get_cluster_output`. This provides access to the cluster's endpoint and certificate data.
   - **Kubeconfig Generation:**  
     A custom kubeconfig is dynamically generated from the cluster information. This configuration is used to initialize a Kubernetes provider, allowing the script to manage Kubernetes resources (like deploying the Helm chart) on your EKS cluster.

2. **OIDC Provider**
   - **Cluster Info and OIDC URL Extraction:**  
     The script uses `aws.eks.get_cluster` to fetch complete cluster details, including identity information. It then extracts the OIDC issuer URL from the cluster's identities.
   - **OIDC Provider Creation:**  
     An OIDC provider is created using the extracted URL, along with required client IDs and a thumbprint. This component is essential for enabling the AWS Load Balancer Controller to assume an IAM role via web identity federation.

3. **IAM Policy for LB Controller**
   - **Policy Loading:**  
     An external IAM policy is loaded from the JSON file (`json/lb-controller-policy.json`).  
   - **Policy Creation:**  
     The policy defines the permissions needed by the AWS Load Balancer Controller to manage AWS resources such as load balancers.

4. **IAM Role with Trust Policy**
   - **Trust Policy Construction:**  
     A custom trust policy is built using the OIDC provider details. This policy specifies that the Kubernetes service account (used by the LB Controller) can assume the role via web identity.
   - **IAM Role Creation:**  
     The IAM role is created using the trust policy. This role is later annotated in the Kubernetes service account, allowing the controller to operate with the required permissions.

5. **IAM Role Policy Attachment**
   - **Policy Attachment:**  
     The IAM policy is attached to the newly created IAM role using an IAM RolePolicyAttachment, ensuring that the role has all necessary permissions.

6. **Helm Chart Deployment for the AWS Load Balancer Controller**
   - **Helm Chart:**  
     The AWS Load Balancer Controller is deployed using a Helm chart sourced from the AWS EKS charts repository.
   - **Chart Configuration:**  
     The chart is configured with key parameters:
     - **Cluster Name and Region:** For correct integration with your EKS cluster.
     - **VPC ID:** Set via the `vpc_id` variable to ensure the load balancer is created in the correct VPC.
     - **Service Account Annotation:** The service account is annotated with the IAM role ARN, enabling the controller to assume the role.


## Deployment
1. **Clone the repository** and navigate into the directory:

   ```sh
   git clone <this-repo-url>
   cd <this-repo-directory>/infrastructure/eks-lb-controller
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
   pulumi stack init rasa-pro-eks-lb-controller
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

| Output Name                  | Description                              |
|------------------------------|------------------------------------------|
| `aws_lb_controller_iam_role` | ARN of the IAM role for the LB Controller|
| `aws_lb_controller_chart`    | Name of the deployed Helm chart          |

## Cleanup

To destroy all resources and remove the infrastructure:

```sh
pulumi destroy
```

To remove the Pulumi stack completely:

```sh
pulumi stack rm [stack_name]
```

## Notes
- The script uses a specific VPC ID to ensure the load balancer is created in the desired VPC
- Ensure your EKS cluster is deployed in the same VPC or adjust the VPC ID accordingly
- The IAM roles, policies, and OIDC provider are set up for the AWS Load Balancer Controller to properly assume the necessary permissions via the Kubernetes service account

## Contributions

Contributions are welcome! Please submit a pull request or open an issue for discussion.