import pulumi
import pulumi_kubernetes as k8s
import pulumi_kubernetes.core.v1 as core
import pulumi_kubernetes.rbac.v1 as rbac
import pulumi_kubernetes.meta.v1 as meta
import pulumi_kubernetes.networking.v1 as networking
import pulumi_kubernetes.helm.v3 as helm
import pulumi_aws as aws
import pulumi.runtime
import os


# Variables
project_name = "rasa-pro"
dashboard_version = "6.0.8" # app version 2.7.0
dashboard_namespace = "kubernetes-dashboard"

# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/rasa-pro-eks-base/ci")
eks_cluster_name = stack_ref.get_output("eks_cluster_name")
vpc_id = stack_ref.get_output("vpc_id")

# DNS and SSL variables
certificate_arn = ""
hosted_zone_id = ""
domain_name = ""

# 📌 **Step 1: Create a Kubernetes Provider**
k8s_provider = k8s.Provider(f"{project_name}-dashboard-k8s-provider")

# 📌 **Step 2: Deploy the Kubernetes Dashboard**
# # Create the "kubernetes-dashboard" namespace
dashboard_ns = core.Namespace(
    dashboard_namespace,
    metadata={"name": dashboard_namespace},
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# Deploy the kubernetes-dashboard Helm chart into the kubernetes-dashboard namespace
dashboard = helm.Chart(
    "kubernetes-dashboard",
    helm.ChartOpts(
        chart="kubernetes-dashboard",
        version=dashboard_version,
        fetch_opts=helm.FetchOpts(
            repo="https://kubernetes.github.io/dashboard/"
        ),
        namespace=dashboard_namespace,
        values={
            "fullnameOverride": "kubernetes-dashboard",
            "replicaCount": 1,
            "service": {
                "port": 443,
                "targetPort": 8443,
                "type": "ClusterIP",
            },
        },
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[dashboard_ns])
)

# 📌 **Step 3: Create a ServiceAccount for Dashboard Admin Access**
dashboard_sa = core.ServiceAccount(
    f"{project_name}-dashboard-sa",
    metadata=meta.ObjectMetaArgs(
        namespace=dashboard_namespace,
        name="dashboard-admin-sa",
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# 📌 **Step 4: Create a Secret to Capture the Dashboard Token**
dashboard_sa_token = core.Secret(
    f"{project_name}-dashboard-token",
    metadata=meta.ObjectMetaArgs(
        namespace=dashboard_namespace,
        name=f"{project_name}-dashboard-token",
        annotations={
            "kubernetes.io/service-account.name": "dashboard-admin-sa",
        }
    ),
    type="kubernetes.io/service-account-token",
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[dashboard_sa])
)

# 📌 **Step 5: Bind the ServiceAccount to the Cluster-Admin Role for Full Access**
dashboard_clusterrolebinding = rbac.ClusterRoleBinding(
    f"{project_name}-dashboard-clusterrolebinding",
    metadata=meta.ObjectMetaArgs(
        name=f"{project_name}-dashboard-clusterrolebinding",
    ),
    role_ref=rbac.RoleRefArgs(
        api_group="rbac.authorization.k8s.io",
        kind="ClusterRole",
        name="cluster-admin",
    ),
    subjects=[{
        "kind": "ServiceAccount",
        "name": "dashboard-admin-sa",
        "namespace": dashboard_namespace,
    }],
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[dashboard_sa])
)

# 📌 **Step 6: Create Ingress for Kubernetes Dashboard to Expose to Load Balancer**
dashboard_ingress = networking.Ingress(
    f"{project_name}-dashboard-ingress",
    metadata=meta.ObjectMetaArgs(
        namespace=dashboard_namespace,
        name="dashboard-ingress",
        annotations={
            "kubernetes.io/ingress.class": "alb",  # Use AWS ALB ingress
            "alb.ingress.kubernetes.io/scheme": "internet-facing",  # Expose to the internet
            "alb.ingress.kubernetes.io/target-type": "ip",  # Target type IP for Kubernetes service
            "alb.ingress.kubernetes.io/listen-ports": '[{"HTTPS": 443}]',  # Listening on port 443
            "alb.ingress.kubernetes.io/backend-protocol": "HTTPS",  # Use HTTPS as Dashboard listens on port 443
            "alb.ingress.kubernetes.io/certificate-arn": certificate_arn,  # ACM certificate ARN for SSL
        }
    ),
    spec=networking.IngressSpecArgs(
        rules=[
            networking.IngressRuleArgs(
                http=networking.HTTPIngressRuleValueArgs(
                    paths=[
                        networking.HTTPIngressPathArgs(
                            path="/",
                            path_type="Prefix",
                            backend=networking.IngressBackendArgs(
                                service=networking.IngressServiceBackendArgs(
                                    name="kubernetes-dashboard",
                                    port=networking.ServiceBackendPortArgs(number=443)
                                )
                            )
                        )
                    ]
                )
            )
        ]
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[dashboard])
)

# 📌 **Step 7: Export the URL for the Load Balancer**
# During preview, the resource may not exist yet, in that case, export a placeholder
if pulumi.runtime.is_dry_run():
    pulumi.export("dashboard_ingress_url", "Preview: Ingress not available")
else:
    # Retrieve the latest state of the Ingress.
    ingress_fresh = networking.Ingress.get(
        "dashboard-ingress-fresh",
        dashboard_ingress.metadata.apply(lambda m: f"{m.namespace}/{m.name}"),
        opts=pulumi.ResourceOptions(provider=k8s_provider)
    )
    dashboard_hostname = ingress_fresh.status.apply(
        lambda status: status.load_balancer.ingress[0].hostname
        if status and status.load_balancer and status.load_balancer.ingress and len(status.load_balancer.ingress) > 0
        else "Ingress not ready"
    )
    pulumi.export("dashboard_ingress_lb_url", dashboard_hostname)

# 📌 **Step 8: Create DNS record for Kubernetes Dashboard**
if not pulumi.runtime.is_dry_run():
    dns_record = aws.route53.Record(
        "dashboard-dns",
        name="dashboard",
        zone_id=hosted_zone_id,
        type="CNAME",
        ttl=300,
        records=[dashboard_hostname],
        opts=pulumi.ResourceOptions(depends_on=[dashboard_ingress])
    )
    pulumi.export("dashboard_url", "dashboard." + domain_name)
