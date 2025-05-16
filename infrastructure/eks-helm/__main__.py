import pulumi
import pulumi_kubernetes as k8s
import pulumi_kubernetes.core.v1 as core
import pulumi_kubernetes.networking.v1 as networking
import pulumi_kubernetes.helm.v3 as helm
import pulumi_kubernetes.meta.v1 as meta
import pulumi_aws as aws
import pulumi.runtime
import os

def get_config(key: str, default: str = None) -> str:
    config = pulumi.Config()
    return config.get(key) or os.getenv(key, default)

# Variables
project_name = get_config("PROJECT_NAME")
pr_name = get_config("PR_NAME")
rasa_pro_helm_version = get_config("RASA_PRO_HELM_VERSION")
rasa_pro_helm_repo = get_config("RASA_PRO_HELM_REPO")
rasa_pro_version = get_config("RASA_PRO_VERSION")
rasa_pro_dev_repository = get_config("RASA_PRO_DEV_REPOSITORY")
rasa_pro_license = get_config("RASA_PRO_LICENSE")
openai_api_key = get_config("OPENAI_API_KEY")
certificate_arn = get_config("CERTIFICATE_ARN")
hosted_zone_id = get_config("HOSTED_ZONE_ID")
domain_name = get_config("DOMAIN_NAME")
# Variables for pulling the model from s3 bucket
model_file_name = get_config("MODEL_FILE_NAME")
aws_model_s3_bucket = get_config("AWS_MODEL_S3_BUCKET")
aws_default_region = get_config("AWS_DEFAULT_REGION")
aws_model_s3_endpoint_url = get_config("AWS_MODEL_S3_ENDPOINT_URL")
aws_model_s3_secret_access_key = get_config("AWS_MODEL_S3_SECRET_ACCESS_KEY")
aws_model_s3_access_key_id = get_config("AWS_MODEL_S3_ACCESS_KEY_ID")


# Retrieve the exported values from the base infrastructure
stack_ref = pulumi.StackReference("rasa/rasa-pro-eks-base/rasa-pro-eks-base")
eks_cluster_name = stack_ref.get_output("eks_cluster_name")
vpc_id = stack_ref.get_output("vpc_id")

# 📌 Step 1: Create a Kubernetes Provider
k8s_provider = k8s.Provider(f"{project_name}-k8s-provider")

# 📌 Step 2: Create the Namespace for Rasa Pro deployment
rasa_pro_ns = core.Namespace(
    project_name,
    metadata={"name": project_name},
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# 📌 Step 2.5: Create the Kubernetes Secret for Rasa Pro
rasa_secrets = core.Secret(
    "rasa-secrets",
    metadata=meta.ObjectMetaArgs(
        name="rasa-secrets",
        namespace=project_name,
    ),
    type="Opaque",
    data={
        # base64-encoded values.
        "rasaProLicense": rasa_pro_license,
        "OpenaiApiKey": openai_api_key,
        "authToken": "",
        "jwtSecret": "",
    },
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[rasa_pro_ns])
)

# 📌 Step 3: Deploy the Rasa Pro Helm Chart from OCI
rasa_pro_chart = helm.Chart(
    "rasa",
    helm.ChartOpts(
        chart=rasa_pro_helm_repo,
        version=rasa_pro_helm_version,
        namespace=project_name,
        values={
            "fullnameOverride": "rasa",
            "service": {
                "type": "LoadBalancer",
                "annotations": {
                    "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
                    "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
                    "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true"
                },
            },
            "rasa": {
                "enabled": True,
                "image": {
                    "tag": rasa_pro_version,
                    "repository": rasa_pro_dev_repository,
                },
                "settings": {
                    "debugMode": True,
                },
                "additionalArgs": ["--model", model_file_name, "--remote-storage", "aws"],
                "additionalEnv": [
                    {"name": "AWS_DEFAULT_REGION", "value": aws_default_region},
                    {"name": "BUCKET_NAME", "value": aws_model_s3_bucket},
                    {"name": "AWS_ENDPOINT_URL", "value": aws_model_s3_endpoint_url},
                    {"name": "AWS_SECRET_ACCESS_KEY", "value": aws_model_s3_secret_access_key},
                    {"name": "AWS_ACCESS_KEY_ID", "value": aws_model_s3_access_key_id},
                    {"name": "OPENAI_API_KEY", "value": openai_api_key},
                ],
                "resources": {
                    "requests": {
                        "cpu": "2",
                        "memory": "3Gi"
                    },
                    "limits": {
                        "cpu": "2",
                        "memory": "3Gi"
                    }
                },
            },
            "rasaProServices": {
                "enabled": False,
            },
        },
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[rasa_pro_ns])
)



# 📌 Step 4: Create Ingress to Expose Rasa Pro via an AWS ALB
rasa_pro_ingress = networking.Ingress(
    f"{project_name}-ingress",
    metadata=meta.ObjectMetaArgs(
        namespace=project_name,
        name="rasa-pro-ingress",
        annotations={
            "kubernetes.io/ingress.class": "alb",  # Use AWS ALB ingress
            "alb.ingress.kubernetes.io/scheme": "internet-facing",  # Expose to the internet
            "alb.ingress.kubernetes.io/target-type": "ip",  # Target type IP for Kubernetes service
            "alb.ingress.kubernetes.io/listen-ports": '[{"HTTPS": 443}]',  # Listen on port 443
            "alb.ingress.kubernetes.io/backend-protocol": "HTTP",  # Backend protocol HTTP
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
                                    name="rasa",  # Matches the fullnameOverride from the chart
                                    port=networking.ServiceBackendPortArgs(number=5005)
                                )
                            )
                        )
                    ]
                )
            )
        ]
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[rasa_pro_chart])
)

# 📌 Step 5: Export the Load Balancer URL for the Ingress
if pulumi.runtime.is_dry_run():
    pulumi.export("rasa_pro_ingress_url", "Preview: Ingress not available")
else:
    ingress_fresh = networking.Ingress.get(
        "rasa-pro-ingress-fresh",
        rasa_pro_ingress.metadata.apply(lambda m: f"{m.namespace}/{m.name}"),
        opts=pulumi.ResourceOptions(provider=k8s_provider)
    )
    ingress_hostname = ingress_fresh.status.apply(
        lambda status: status.load_balancer.ingress[0].hostname
        if status and status.load_balancer and status.load_balancer.ingress and len(status.load_balancer.ingress) > 0
        else "Ingress not ready"
    )
    pulumi.export("rasa_pro_ingress_lb_url", ingress_hostname)

# 📌 Step 6: Create a DNS Record for the Rasa Pro Ingress
if not pulumi.runtime.is_dry_run():
    dns_record = aws.route53.Record(
        project_name + "-dns",
        name=pr_name,
        zone_id=hosted_zone_id,
        type="CNAME",
        ttl=300,
        records=[ingress_hostname],
        opts=pulumi.ResourceOptions(depends_on=[rasa_pro_ingress])
    )
    pulumi.export("rasa_pro_url", project_name + "." + domain_name)
