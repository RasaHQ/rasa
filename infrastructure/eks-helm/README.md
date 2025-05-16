# Rasa Pro Helm Deployment on EKS (Fargate)

## Overview

This Pulumi script deploys **Rasa Pro** onto an existing AWS EKS cluster using Fargate, a Helm chart, and an AWS Application Load Balancer (ALB). It includes:

- Kubernetes namespace creation
- Helm chart deployment of Rasa Pro
- Secure environment variable injection via Kubernetes secrets
- ALB ingress setup with TLS termination
- Route53 DNS record creation
- S3 model file integration

> **ℹ️ This deployment script is designed to run only via GitHub Actions on pull requests.**
>
> It automates the deployment of a dev version of Rasa Pro into a temporary namespace on CI EKS cluster for integration testing purposes.
> It is not intended for manual execution or production use.

---

## Resources
```
     Type                                        Name                                      Plan       Info
     pulumi:pulumi:Stack                         rasa-pro-eks-helm-pr-2311                            
     ├─ kubernetes:helm.sh/v3:Chart              rasa                                                 
     │  ├─ kubernetes:core/v1:ServiceAccount     rasa-pro-pr-2311/rasa                                
     │  ├─ kubernetes:core/v1:Service            rasa-pro-pr-2311/rasa                                
     │  ├─ kubernetes:core/v1:ConfigMap          rasa-pro-pr-2311/rasa-configmap                      
     │  └─ kubernetes:apps/v1:Deployment         rasa-pro-pr-2311/rasa                                
     ├─ pulumi:pulumi:StackReference             rasa/rasa-pro-eks-base/rasa-pro-eks-base             
     ├─ kubernetes:networking.k8s.io/v1:Ingress  rasa-pro-ingress-fresh                         
     ├─ kubernetes:core/v1:Secret                rasa-secrets                                         
     ├─ kubernetes:core/v1:Namespace             rasa-pro-pr-2311                                     
     ├─ aws:route53:Record                       rasa-pro-pr-2311-dns                                 
     ├─ pulumi:providers:kubernetes              rasa-pro-pr-2311-k8s-provider                        
     └─ kubernetes:networking.k8s.io/v1:Ingress  rasa-pro-pr-2311-ingress                             

+ 12
```

---

## Prerequisites

Ensure the following are installed and configured:

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- [AWS CLI](https://aws.amazon.com/cli/)
- Python 3.x with `pip`
- A deployed EKS base stack (`rasa/rasa-pro-eks-base/rasa-pro-eks-base`)
- Required config values set as Pulumi config or environment variables

---

## Configuration Variables

The script expects these config values via environment variables:

| Key                              | Description                                |
|----------------------------------|--------------------------------------------|
| `PROJECT_NAME`                   | Namespace and resource prefix              |
| `RASA_PRO_HELM_VERSION`          | Helm chart version                         |
| `RASA_PRO_HELM_REPO`             | OCI chart reference                        |
| `RASA_PRO_VERSION`               | Docker image tag to deploy                 |
| `RASA_PRO_DEV_REPOSITORY`        | Docker image repository                    |
| `RASA_PRO_LICENSE`               | License key for Rasa Pro                   |
| `OPENAI_API_KEY`                 | OpenAI API Key (if using LLM features)     |
| `CERTIFICATE_ARN`                | ACM TLS certificate ARN                    |
| `HOSTED_ZONE_ID`                 | Route53 hosted zone ID                     |
| `DOMAIN_NAME`                    | Domain to map (e.g., `rasapro.rasa.io`)    |
| `MODEL_FILE_NAME`                | Name of model file in S3                   |
| `AWS_MODEL_S3_BUCKET`            | S3 bucket name for models                  |
| `AWS_MODEL_S3_ENDPOINT_URL`      | S3 endpoint URL                            |
| `AWS_MODEL_S3_SECRET_ACCESS_KEY` | S3 secret key                              |
| `AWS_MODEL_S3_ACCESS_KEY_ID`     | S3 access key                              |
| `AWS_DEFAULT_REGION`             | AWS region (e.g., `eu-west-1`)             |

---

## Cleanup

All resources provisioned by this deployment are automatically deleted when the corresponding pull request is closed.
This ensures no leftover infrastructure remains.

---

## Notes
- The ALB is internet-facing and terminates TLS using the provided ACM certificate.
- Kubernetes pods are deployed on AWS Fargate, no EC2 nodes are required.
- The Rasa Pro container uses custom dev image pulled from ECR and loads the trained model from S3.

---

## Contributions

Contributions are welcome! Please submit a pull request or open an issue for discussion.