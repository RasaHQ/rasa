# Headlamp Deployment with Pulumi

This project deploys [Headlamp](https://headlamp.dev/) onto an AWS EKS cluster using Pulumi and a Helm chart. It installs Headlamp into the specified namespace, sets up an AWS ALB ingress, and creates a Route 53 DNS record to expose the UI externally.

## Overview

The Pulumi script performs the following steps:

1. **Kubernetes Provider Setup:**  
   Configures a Pulumi Kubernetes provider targeting your EKS cluster.

2. **Helm Chart Deployment:**  
   Installs the Headlamp Helm chart with these customizations:
   - Chart version defined by `headlamp_version` (e.g. `0.30.0`).
   - Overrides the full‑name to `headlamp`.
   - Sets `replicaCount: 1` and a `ClusterIP` service on port 80.

3. **Ingress Configuration:**  
   Creates a Kubernetes `Ingress` resource annotated for the AWS Load Balancer Controller:
   - Internet‑facing ALB listening on HTTP (80) and HTTPS (443).
   - Uses ACM certificate for TLS.
   - Routes `/` to the Headlamp service on port 80.

4. **Load Balancer URL Export:**  
   Exports the ALB hostname as `headlamp_ingress_lb_url`.

5. **DNS Record Creation:**  
   Creates a CNAME record (`headlamp.<domain_name>`) in Route 53 pointing to the ALB.

## Resources

A typical `pulumi up` will show resources like:

```
     Type                                                              Name                            Status
 +   pulumi:pulumi:Stack                                               eks-headlamp-ci                 created (55s)       
 +   ├─ kubernetes:helm.sh/v3:Chart                                    headlamp                        created (55s)       
 +   │  ├─ kubernetes:apps/v1:Deployment                               kube-system/headlamp            created (51s)       
 +   │  ├─ kubernetes:core/v1:ServiceAccount                           kube-system/headlamp            created (0.00s)     
 +   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRoleBinding  headlamp-admin                  created (0.00s)     
 +   │  ├─ kubernetes:core/v1:Secret                                   kube-system/oidc                created (0.00s)     
 +   │  └─ kubernetes:core/v1:Service                                  kube-system/headlamp            created (12s)       
 +   ├─ aws:route53:Record                                             headlamp-dns                    created (36s)       
 +   ├─ pulumi:providers:kubernetes                                    rasa-pro-headlamp-k8s-provider  created (0.00s)     
 +   └─ kubernetes:networking.k8s.io/v1:Ingress                        rasa-pro-headlamp-ingress       created (4s)        

 + 10 created
```  

## Prerequisites

- **AWS EKS Cluster:** An existing EKS cluster stack reference `rasa/rasa-pro-eks-base/ci` exporting `eks_cluster_name` and `vpc_id`.
- **Pulumi CLI:** Installed and logged in.
- **Python 3.8+** with `pulumi`, `pulumi_kubernetes`, and `pulumi_aws` packages.
- **AWS CLI & Credentials:** Permissions for EKS, ALB creation, ACM, and Route 53.
- **AWS Load Balancer Controller** installed in your cluster.
- **SSL & DNS:**  
  - An ACM certificate ARN for `headlamp.<domain_name>`.  
  - A Route 53 hosted zone ID.  
  - Your root domain (e.g., `rasapro.rasa-dev.io`).

## Configuration

Update these variables at the top of the Pulumi script or via `pulumi config`:

| Variable               | Description                                  | Example                          |
|------------------------|----------------------------------------------|----------------------------------|
| `project_name`         | Prefix for all resources                     | `rasa-pro`                       |
| `headlamp_namespace`   | Kubernetes namespace for Headlamp            | `kube-system`                    |
| `headlamp_version`     | Helm chart version for Headlamp              | `0.30.0`                         |
| `certificate_arn`      | ACM certificate ARN for TLS                  | `arn:aws:acm:...:certificate/...`|
| `hosted_zone_id`       | Route 53 hosted zone ID                      | `Your hosted zone ID`            |
| `domain_name`          | Root domain for DNS record                   | `rasapro.rasa-dev.io`            |

## Deployment

1. **Select or create** your Pulumi stack:
   ```bash
   pulumi stack init <org>/rasa-pro-headlamp/dev
   pulumi stack select dev
   ```

2. **Set configuration** (if not using inline variables):
   ```bash
   pulumi config set project_name "rasa-pro"
   pulumi config set headlamp_namespace "kube-system"
   pulumi config set headlamp_version "0.30.0"
   pulumi config set certificate_arn <YOUR_CERT_ARN>
   pulumi config set hosted_zone_id <YOUR_ZONE_ID>
   pulumi config set domain_name "rasapro.rasa-dev.io"
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Preview and apply**:
   ```bash
   pulumi preview
   pulumi up --yes
   ```

## Outputs

| Output Name                 | Description                              |
|-----------------------------|------------------------------------------|
| `headlamp_ingress_lb_url`   | The ALB hostname for Headlamp UI         |
| `headlamp_url`              | The CNAME (`headlamp.<domain_name>`)     |

## Cleanup

To destroy resources:
```bash
pulumi destroy --yes
```
To remove the stack:
```bash
pulumi stack rm dev --yes
```

## Notes

### VPC & Subnet Tags

The AWS Load Balancer Controller discovers networking resources via tags:

- **VPC Tag:**  
  `Key: kubernetes.io/cluster/<cluster-name>`,  
  `Value: shared` or `owned`  
- **Subnet Tags for Internet‑Facing ALB:**  
  `Key: kubernetes.io/role/elb`,  
  `Value: 1`

## Contributions

Contributions welcome! Please open an issue or pull request.

