import pulumi
import pulumi_kubernetes as k8s
import pulumi_kubernetes.core.v1 as core
import pulumi_kubernetes.meta.v1 as meta
import pulumi_kubernetes.networking.v1 as networking
import pulumi_kubernetes.helm.v3 as helm
import pulumi_aws as aws
import pulumi.runtime
import os

# Variables
project_name = "rasa-pro"
headlamp_namespace = "kube-system"
headlamp_version = "0.30.0"

# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/rasa-pro-eks-base/ci")
eks_cluster_name = stack_ref.get_output("eks_cluster_name")
vpc_id = stack_ref.get_output("vpc_id")

# DNS and SSL variables
certificate_arn = ""
hosted_zone_id = ""
domain_name = "rasapro.rasa-dev.io"

# 📌 Step 1: Create a Kubernetes Provider
k8s_provider = k8s.Provider(f"{project_name}-headlamp-k8s-provider")

# 📌 Step 2: Deploy Headlamp via the Helm chart
headlamp = helm.Chart(
    "headlamp",
    helm.ChartOpts(
        chart="headlamp",
        version=headlamp_version,
        fetch_opts=helm.FetchOpts(
            repo="https://kubernetes-sigs.github.io/headlamp/"
        ),
        namespace=headlamp_namespace,
        values={
            # Override the full name so that resources are named "headlamp"
            "fullnameOverride": "headlamp",
            "replicaCount": 1,
            "service": {
                "type": "ClusterIP",
                "port": 80,
            },
        },
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# 📌 Step 3: Create an Ingress to expose Headlamp via an AWS ALB
headlamp_ingress = networking.Ingress(
    f"{project_name}-headlamp-ingress",
    metadata=meta.ObjectMetaArgs(
        namespace=headlamp_namespace,
        name="headlamp-ingress",
        annotations={
            "kubernetes.io/ingress.class": "alb",
            "alb.ingress.kubernetes.io/scheme": "internet-facing",
            "alb.ingress.kubernetes.io/target-type": "ip",
            "alb.ingress.kubernetes.io/listen-ports": '[{"HTTP":80}, {"HTTPS":443}]',
            "alb.ingress.kubernetes.io/backend-protocol": "HTTP",
            "alb.ingress.kubernetes.io/certificate-arn": certificate_arn,
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
                                    name="headlamp",
                                    port=networking.ServiceBackendPortArgs(number=80)
                                )
                            )
                        )
                    ]
                )
            )
        ]
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[headlamp])
)

# 📌 Step 4: Export the Load Balancer URL for the Ingress
if pulumi.runtime.is_dry_run():
    pulumi.export("headlamp_ingress_url", "Preview: Ingress not available")
else:
    ingress_fresh = networking.Ingress.get(
        "headlamp-ingress-fresh",
        headlamp_ingress.metadata.apply(lambda m: f"{m.namespace}/{m.name}"),
        opts=pulumi.ResourceOptions(provider=k8s_provider)
    )
    headlamp_hostname = ingress_fresh.status.apply(
        lambda status: status.load_balancer.ingress[0].hostname
        if status and status.load_balancer and status.load_balancer.ingress and len(status.load_balancer.ingress) > 0
        else "Ingress not ready"
    )
    pulumi.export("headlamp_ingress_lb_url", headlamp_hostname)

# 📌 Step 5: Create a DNS record for Headlamp
if not pulumi.runtime.is_dry_run():
    dns_record = aws.route53.Record(
        "headlamp-dns",
        name="headlamp",
        zone_id=hosted_zone_id,
        type="CNAME",
        ttl=300,
        records=[headlamp_hostname],
        opts=pulumi.ResourceOptions(depends_on=[headlamp_ingress])
    )
    pulumi.export("headlamp_url", "headlamp." + domain_name)
