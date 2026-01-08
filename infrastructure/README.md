# Infrastructure Architecture

The infrastructure setup is divided into separate Pulumi projects (micro-stacks):

- **eks-helm**
- **eks-integration-tests**

## 1. eks-helm

The **eks-helm** stack is responsible for deploying the **Rasa Pro application container** using the **Rasa Pro Helm chart** into the EKS stack created in **eks-base**, with the **load balancing resources & Fargate profile** created in the **eks-lb-controller** stack.

The official **Rasa Pro Helm chart** is maintained [here](#).

The released **Rasa Pro Helm chart** can be pulled from:
```
oci://europe-west3-docker.pkg.dev/rasa-releases/helm-charts/rasa
```

## 2. eks-integration-tests

The **eks-integration-tests** stack is responsible for creating and managing all infrastructure resources required for integration testing. This includes:

- **RDS Database**: PostgreSQL database instance used for Tracker Store testing
- **Pulumi Secrets**: Database credentials managed through Pulumi's built-in secrets management system
- **VPC Integration**: Database deployed within the same VPC as the EKS cluster for secure connectivity

This infrastructure provides a dedicated testing environment with persistent database resources that can be used across multiple integration test runs, ensuring reliable and consistent testing conditions for Rasa Pro components.
