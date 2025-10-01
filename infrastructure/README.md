# Infrastructure Architecture

The infrastructure setup is divided into six separate Pulumi projects (micro-stacks):

- **eks-base**
- **eks-lb-controller**
- **eks-dashboard**
- **eks-headlamp**
- **eks-helm**
- **eks-integration-tests**

## 1. eks-base

The infra base project is responsible for creating:

- **Network** for housing Rasa Pro deployments (VPC, Subnets, NAT gateway, internet gateway)
- **EKS Cluster** (Kubernetes cluster on AWS using AWS Elastic Kubernetes Service) to support all deployments.

The resources created in this stack are then used in the **eks-lb-controller** and **eks-rasa-pro-helm** stacks.

Pull request deployments are **not enabled** here since these components rarely change. All pull request deployments will share the resources created by the **eks-base** project. Thus, the **eks-base** stack creation is a one-time setup run locally from a developer machine by configuring the required variables in the `.env` file.

## 2. eks-lb-controller

The **EKS Load Balancer Controller** is responsible for creating a common **AWS Load Balancer Controller**, which is shared between all Rasa Pro application deployments under the EKS cluster. This project also creates a **Fargate profile** to provision compute instances for all deployments powered by **AWS Fargate**.

The **AWS LB Controller** takes care of creating **AWS Application Load Balancers (ALBs)** for the ingress resources created during deployment. To optimize usage and reduce cost, we use a **single ALB** for all Rasa Pro app deployments. Routing to different applications happens via the host URL. For example:

- `http://pr580.rasapro.rasa-dev.io` routes users to the ingress pointing to the `pr580` deployment.
- `http://pr581.rasapro.rasa-dev.io` routes users to the ingress pointing to the `pr581` deployment.

Both deployments use a **shared load balancer**, optimizing cost and deployment speed.

Pull request deployments are **not enabled** here since these components rarely change. All pull request deployments will share the resources created by the **eks-lb-controller** project. Thus, the **eks-lb-controller** stack creation is a **one-time setup**, run locally from a developer machine by setting up the required variables in the `.env` file.

### Logging

EKS has an **Amazon CloudWatch Observability** add-on, which automatically takes care of logging.

## 3. eks-dashboard
The **EKS Dashboard** stack is responsible for deploying Kubernetes Dashboard.  It creates all necessary Kubernetes resources - including a namespace, service account with admin privileges, token secret, cluster role binding, and an ingress configured with an AWS Application Load Balancer (ALB). Additionally, it creates a DNS record in AWS Route53 to expose the dashboard externally.

## 4. eks-headlamp
The **EKS Headlamp** stack is responsible for deploying Headlamp, a modern Kubernetes dashboard and cluster management tool. It creates all necessary Kubernetes resources including a namespace, service account, cluster role binding, and an ingress configured with an AWS Application Load Balancer (ALB). Additionally, it creates a DNS record in AWS Route53 to expose the Headlamp interface externally.

## 5. eks-helm

The **eks-helm** stack is responsible for deploying the **Rasa Pro application container** using the **Rasa Pro Helm chart** into the EKS stack created in **eks-base**, with the **load balancing resources & Fargate profile** created in the **eks-lb-controller** stack.

The official **Rasa Pro Helm chart** is maintained [here](#).

The released **Rasa Pro Helm chart** can be pulled from:
```
oci://europe-west3-docker.pkg.dev/rasa-releases/helm-charts/rasa
```

### Pull Request Deployments

For every **Pull Request** created in the **Rasa Private repo**, a new deployment of the **eks-rasa-pro-helm** is initiated on the common **EKS cluster**.

## 6. eks-integration-tests

The **eks-integration-tests** stack is responsible for creating and managing all infrastructure resources required for integration testing. This includes:

- **RDS Database**: PostgreSQL database instance used for Tracker Store testing
- **Pulumi Secrets**: Database credentials managed through Pulumi's built-in secrets management system
- **VPC Integration**: Database deployed within the same VPC as the EKS cluster for secure connectivity

This infrastructure provides a dedicated testing environment with persistent database resources that can be used across multiple integration test runs, ensuring reliable and consistent testing conditions for Rasa Pro components.
