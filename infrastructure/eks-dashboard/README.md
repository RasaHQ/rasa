# Kubernetes Dashboard Deployment with Pulumi

This project deploys the Kubernetes Dashboard onto an AWS EKS cluster using Pulumi and a Helm chart. It creates all necessary Kubernetes resources—including a namespace, service account with admin privileges, token secret, cluster role binding, and an ingress configured with an AWS Application Load Balancer (ALB). Additionally, it creates a DNS record in AWS Route53 to expose the dashboard externally.

## Overview

The Pulumi script performs the following steps:

1. **Kubernetes Provider Setup:**  
   Configures a Pulumi Kubernetes provider to target your EKS cluster.

2. **Namespace Creation:**  
   Creates the `kubernetes-dashboard` namespace where the dashboard will be deployed.

3. **Helm Chart Deployment:**  
   Deploys the Kubernetes Dashboard using a Helm chart with the following customizations:
   - Sets the Helm chart version to `6.0.8` (which corresponds to Dashboard app version 2.7.0).
   - Overrides the full name to `kubernetes-dashboard`.
   - Configures the service to expose port 443 (mapped to container port 8443).
   - Sets the replica count to 1.

4. **Service Account and Token:**  
   Creates a service account (`dashboard-admin-sa`) and a corresponding secret to store the dashboard token.

5. **Cluster Role Binding:**  
   Binds the service account to the `cluster-admin` ClusterRole to grant full access.

6. **Ingress Configuration:**  
   Creates an ingress resource that:
   - Uses the AWS ALB ingress controller.
   - Listens on HTTPS (port 443) using an ACM certificate.
   - Routes external traffic to the Kubernetes Dashboard service.

7. **DNS Record Creation:**  
   Creates a DNS record in AWS Route53 so that the dashboard is accessible via a friendly URL (e.g., `dashboard.<domain_name>`).

## Resources
```
   Type                                                              Name                                                  Status       
+   pulumi:pulumi:Stack                                               rasa-pro-eks-dashboard-rasa-pro-kubernetes-dashboard  created     
+   ├─ kubernetes:helm.sh/v3:Chart                                    kubernetes-dashboard                                  created     
+   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRoleBinding  kubernetes-dashboard-metrics                          created     
+   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRole         kubernetes-dashboard-metrics                          created     
+   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:RoleBinding         kubernetes-dashboard/kubernetes-dashboard             created     
+   │  ├─ kubernetes:core/v1:Secret                                   kubernetes-dashboard/kubernetes-dashboard-certs       created     
+   │  ├─ kubernetes:core/v1:Secret                                   kubernetes-dashboard/kubernetes-dashboard-key-holder  created     
+   │  ├─ kubernetes:core/v1:Service                                  kubernetes-dashboard/kubernetes-dashboard             created     
+   │  ├─ kubernetes:rbac.authorization.k8s.io/v1:Role                kubernetes-dashboard/kubernetes-dashboard             created     
+   │  ├─ kubernetes:core/v1:ServiceAccount                           kubernetes-dashboard/kubernetes-dashboard             created     
+   │  ├─ kubernetes:core/v1:Secret                                   kubernetes-dashboard/kubernetes-dashboard-csrf        created     
+   │  ├─ kubernetes:apps/v1:Deployment                               kubernetes-dashboard/kubernetes-dashboard             created     
+   │  └─ kubernetes:core/v1:ConfigMap                                kubernetes-dashboard/kubernetes-dashboard-settings    created     
+   ├─ aws:route53:Record                                             dashboard-dns                                         created     
+   ├─ kubernetes:networking.k8s.io/v1:Ingress                        rasa-pro-dashboard-ingress                            created     
+   ├─ kubernetes:rbac.authorization.k8s.io/v1:ClusterRoleBinding     rasa-pro-dashboard-clusterrolebinding                 created     
+   ├─ kubernetes:core/v1:ServiceAccount                              rasa-pro-dashboard-sa                                 created     
+   ├─ kubernetes:core/v1:Namespace                                   kubernetes-dashboard                                  created     
+   ├─ kubernetes:core/v1:Secret                                      rasa-pro-dashboard-token                              created     
+   └─ pulumi:providers:kubernetes                                    rasa-pro-dashboard-k8s-provider                       created 

+ 20
```

## Prerequisites

- **AWS EKS Cluster:** An existing EKS cluster (`rasa/rasa-pro-eks-base/rasa-pro-eks-base`)
- **Pulumi CLI:** Installed and configured (see [Pulumi Installation](https://www.pulumi.com/docs/get-started/install/))
- **Python:** Version 3.8 or higher. 
- **AWS CLI & Credentials:** Configured for your AWS account with appropriate permissions (EKS, ACM, Route53).
- **Load Balancer Controller** (eks-lb-controller)
- **SSL & DNS:**  
  - An ACM certificate ARN for your dashboard domain
  - A Route53 hosted zone ID for your domain
  - Your domain name (e.g., `rasapro.rasa-dev.io`)

## Configuration

Edit the following variables in the script as needed:

- `project_name`: The project name (default is `"rasa-pro"`).
- `dashboard_version`: The Helm chart version (set to `"6.0.8"` for app version 2.7.0).
- `dashboard_namespace`: Namespace for the dashboard (`"kubernetes-dashboard"`).
- **DNS and SSL Variables:**  
  - `certificate_arn`: Your ACM certificate ARN.
  - `hosted_zone_id`: Your Route53 hosted zone ID.
  - `domain_name`: Your domain name.

## Deployment

1. **Clone the repository** and navigate into the directory:

   ```sh
   git clone <this-repo-url>
   cd <this-repo-directory>/infrastructure/eks-dashboard
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
   pulumi stack init rasa-pro-eks-dashboard
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

| Output Name                | Description                   |
|----------------------------|-------------------------------|
| `dashboard_ingress_lb_url` | Load balancer’s URL           |
| `dashboard_url`            | Domain URL                    |


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
### VPC and Subnets tag 

Tags are a key part of the discovery process that allows the AWS Load Balancer Controller to automatically configure and manage your ALB or NLB resources without having to manually specify every networking detail.

VPC and Subnets tag configuration:

* VPC:
    * Key: kubernetes.io/cluster/rasa-pro-cluster
    * Value: shared (or owned if the VPC is exclusively used by your cluster)
* Internet-Facing Load Balancers:
    * Key: kubernetes.io/role/elb
    * Value: 1
* Internal Load Balancers:
    * Key: kubernetes.io/role/internal-elb
    * Value: 1

### Kubernetes Dashboard token

Since kubernetes dashboard token is sensitive, not suitable for export, it can be retrived using:
`kubectl get secret -n kubernetes-dashboard rasa-pro-dashboard-token -o jsonpath='{.data.token}' | base64 --decode`

## Contributions

Contributions are welcome! Please submit a pull request or open an issue for discussion.
