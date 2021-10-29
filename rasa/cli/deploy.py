import argparse
import base64
import copy
from distutils.dir_util import copy_tree
import hashlib
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Text, Tuple

from google.api_core.exceptions import ClientError
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient import discovery
import pkg_resources
import questionary
from questionary import Choice
import requests
from tqdm import tqdm

from rasa.cli.arguments import deploy as arguments
from rasa.cli.utils import print_error, print_info, print_success
from rasa.constants import PACKAGE_NAME
from rasa.model import get_latest_model
from rasa.nlu.utils import write_json_to_file
from rasa.utils.io import read_file, read_yaml_file, write_text_file, write_yaml_file

logger = logging.getLogger(__name__)


PACKAGE_NAME_DOCKER_ACTIONS = "rasa-actions"
PACKAGE_NAME_DOCKER_HELM = "helm"
PACKAGE_NAME_DOCKER_RASA = "rasa-helm"
PACKAGE_NAME_DOCKER_TERRAFORRM = "rasa-terraform"

DOCKERFILE_TEMPLATE = "cli/Dockerfile_default"

REQUIRED_GOOGLE_CLOUD_SERVICES = [
    "cloudbuild.googleapis.com",
    "storage-api.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "cloudresourcemanager.googleapis.com",
]


def sha256sum_tar(filename: Text) -> Text:
    tar_hash = hashlib.sha256()

    with tarfile.open(filename, "r") as tar:
        for tarinfo in tar:
            if not tarinfo.isreg():
                continue

            flo = tar.extractfile(tarinfo)

            if not flo:
                continue

            while True:
                data = flo.read(2 ** 20)
                if not data:
                    break
                tar_hash.update(data)

            flo.close()

    return tar_hash.hexdigest()


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    channels_parser = subparsers.add_parser(
        "deploy",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Deploy your assistant to the cloud.",
    )
    channels_parser.set_defaults(func=configure_deployment)

    arguments.set_deploy_arguments(channels_parser)


def get_service_account(args: argparse.Namespace, project_name: Text) -> Text:
    if args.service_account:
        return args.service_account
    else:
        return ask_iam_service_account(project_name)


def ask_iam_service_account(project_name: Text) -> Text:
    print_info(
        f"Now you need to create a service account, to allow Rasa to "
        f"create the necessary resources in the project. The service account needs to"
        f"have OWNER permissions and can have an arbitrary name! "
        f"You can create the account at "
        f"https://console.cloud.google.com/iam-admin/serviceaccounts/create?project={project_name}"
    )

    if not questionary.confirm("Done?").ask():
        print_error(":(")
        sys.exit(1)

    print_info(
        f"Now you need to create a JSON key for that service account at "
        f"https://console.cloud.google.com/iam-admin/serviceaccounts?project={project_name} "
    )
    key_path = questionary.text(
        "What's the path of the key file for the service account"
    ).ask()

    if not key_path:
        print_error(":(")
        sys.exit(1)

    return key_path


def get_project_name(args: argparse.Namespace) -> Text:
    if args.project_id:
        return args.project_id
    else:
        return ask_project_name()


def ask_project_name() -> Text:
    print_info(
        "For the deployment we'll need a google cloud project. "
        "You can use an existing one (DONT DO IT, THIS IS VERY ALPHA) or "
        "create a new project at https://console.cloud.google.com/projectcreate"
    )

    project_name = questionary.text(
        "What is the Project ID of your google cloud project?"
    ).ask()
    if not project_name:
        print_error(":(")
        sys.exit(1)

    return project_name


def exclude_dot_files(tar_info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if Path(tar_info.name).stem.startswith("."):
        logger.debug(f"Excluded file {tar_info.name} from tar.")
        return None
    else:
        return tar_info


def create_temporary_tar(
    file_or_directory: Text, arcname: Optional[Text] = None
) -> Text:
    file = tempfile.NamedTemporaryFile(suffix=".tgz", delete=False)
    with tarfile.open(file.name, "w:gz") as tar:

        # if we are packaging a directory we need to properly set the base name in the
        # archive
        if arcname is None and os.path.isdir(file_or_directory):
            arcname = os.path.basename(file_or_directory)

        tar.add(file_or_directory, arcname=arcname, filter=exclude_dot_files)
    return file.name


def create_terraform_vars(path: Text, data: Dict[Text, Any]) -> None:
    write_json_to_file(os.path.join(path, "terraform.tfvars.json"), data)


def create_storage_bucket(
    project_name: Text, service_account_file: Text, bucket_name: Text
) -> None:
    def _does_bucket_exist(client):
        try:
            return client.bucket(bucket_name).exists()
        except ClientError:
            return False

    storage_client = storage.Client.from_service_account_json(
        service_account_file, project=project_name
    )

    if not _does_bucket_exist(storage_client):
        storage_client.create_bucket(bucket_name)


def upload_build_context(
    project_name: Text,
    service_account_file: Text,
    package_name: Text,
    context_tar: Text,
) -> Tuple[Text, Text]:
    storage_client = storage.Client.from_service_account_json(
        service_account_file, project=project_name
    )

    context_hash = sha256sum_tar(context_tar)

    bucket_name = f"{project_name}_{package_name}_cloudbuild"
    create_storage_bucket(project_name, service_account_file, bucket_name)

    file_key = f"{context_hash}.tgz"

    logger.debug(f"Checking for context hash '{context_hash}' in bucket ")
    blob = storage_client.bucket(bucket_name).blob(file_key)

    if not blob.exists():
        logger.info(f"Uploading {package_name} to google cloud storage")
        blob.upload_from_filename(context_tar)
    else:
        logger.debug(f"Build context already exists, reusing it.")

    return bucket_name, file_key


def trigger_docker_build(
    project_name: Text,
    service_account_file: Text,
    package_name: Text,
    tag: Text,
    context_tar: Text,
    preprocessing_steps: Optional[List[Dict[Text, Any]]] = None,
) -> Text:
    logger.debug(f"Starting docker build for {package_name}")

    image_name = f"gcr.io/{project_name}/{package_name}:{tag}"

    if not preprocessing_steps:
        preprocessing_steps = []

    steps = preprocessing_steps + [
        {
            "args": [
                "build",
                "--network",
                "cloudbuild",
                "--no-cache",
                "-t",
                image_name,
                ".",
            ],
            "name": "gcr.io/cloud-builders/docker",
        }
    ]

    build_id = trigger_cloud_build(
        project_name,
        service_account_file,
        package_name,
        context_tar,
        steps,
        image_names=[image_name],
    )
    return build_id


def get_enabled_services(project_name: Text, service_client: Any) -> Set[Text]:
    response = (
        service_client.services()
        .list(parent=f"projects/{project_name}", filter="state:ENABLED")
        .execute()
    )

    return {
        service.get("config", {}).get("name")
        for service in response.get("services", [])
    }


def enable_gcloud_services(
    project_name: Text, service_account_file: Text, service_ids: List[Text]
) -> None:
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )

    service_client = discovery.build(
        "serviceusage", "v1", credentials=credentials, cache_discovery=False
    )

    enabled_services = get_enabled_services(project_name, service_client)
    inactive_services = [s for s in service_ids if s not in enabled_services]

    if not inactive_services:
        logger.debug("Nothing to enable, all services are already enabled.")
        return

    logger.info(f"Trying to enable google API services {', '.join(inactive_services)}")
    request_body = {"serviceIds": inactive_services}

    operation = (
        service_client.services()
        .batchEnable(parent=f"projects/{project_name}", body=request_body)
        .execute()
    )

    is_done = False

    with tqdm(desc="Enabling google APIs") as pbar:
        while not is_done:
            response = service_client.operations().get(name=operation["name"]).execute()
            is_done = response.get("done", False)
            if not is_done:
                time.sleep(5)
            pbar.refresh()

        # sometimes it still takes a bit longer, so let's wait until they
        # show up as enabled
        while not all(inactive in enabled_services for inactive in inactive_services):
            enabled_services = get_enabled_services(project_name, service_client)
            time.sleep(3)
            pbar.refresh()

    print_success(f"Finished enabling google cloud APIs!")


def trigger_cloud_build(
    project_name: Text,
    service_account_file: Text,
    package_name: Text,
    context_tar: Text,
    steps: List[Dict[Text, Any]],
    image_names: Optional[List[Text]] = None,
) -> Text:
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )

    build_client = discovery.build(
        "cloudbuild", "v1", credentials=credentials, cache_discovery=False
    )

    bucket_name, file_key = upload_build_context(
        project_name, service_account_file, package_name, context_tar
    )

    body = {
        "projectId": project_name,
        "source": {"storageSource": {"bucket": bucket_name, "object": file_key,}},
        "steps": steps,
        "images": [image_names or []],
    }
    response = (
        build_client.projects()
        .builds()
        .create(projectId=project_name, body=body)
        .execute()
    )
    build_id = response.get("metadata", {}).get("build", {}).get("id")
    return build_id


def check_build(
    project_name: Text, build_id: Text, service_account_file: Text
) -> Dict[Text, Any]:
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )
    build_client = discovery.build(
        "cloudbuild", "v1", credentials=credentials, cache_discovery=False
    )

    return (
        build_client.projects()
        .builds()
        .get(projectId=project_name, id=build_id)
        .execute()
    )


def wait_for_build_to_finish(
    project_name: Text, build_id: Text, service_account_file: Text, package_name: Text
) -> None:
    print(
        f"You can follow the detailed log at "
        f"https://console.cloud.google.com/cloud-build/builds/{build_id}"
    )

    with tqdm(desc=f"Building image {package_name}") as pbar:
        while True:
            response = check_build(project_name, build_id, service_account_file)
            build_status = response.get("status")
            pbar.refresh()

            if build_status in {"QUEUED", "WORKING"}:
                time.sleep(5)
            else:
                break

    if build_status == "SUCCESS":
        logger.debug(f"Image build exited with {build_status}...")

    else:
        build_log_url = response.get("logUrl")
        print_error(f"Build failed, aborting :(")

        if build_log_url:
            print_error(
                f"You can find the full cloud build log with "
                f"the error at {build_log_url}"
            )
        sys.exit(1)


def build_actions_docker(project_name: Text, service_account_file: Text) -> None:
    print_info("Preparing your action server docker container...")
    create_dockerfile("Dockerfile")

    tar_path = create_temporary_tar(".")
    logger.debug(f"Compressed project to {tar_path}")
    build_id = trigger_docker_build(
        project_name,
        service_account_file,
        package_name=PACKAGE_NAME_DOCKER_ACTIONS,
        tag="latest",
        context_tar=tar_path,
    )

    wait_for_build_to_finish(
        project_name, build_id, service_account_file, PACKAGE_NAME_DOCKER_ACTIONS
    )


def build_helm_docker(project_name: Text, service_account_file: Text) -> None:
    print_info("Preparing HELM...")
    file = tempfile.NamedTemporaryFile(suffix=".random", delete=False)
    tar_path = create_temporary_tar(file.name)

    steps = [
        {
            "id": "clone helm builder",
            "name": "gcr.io/cloud-builders/git",
            "args": [
                "clone",
                "https://github.com/GoogleCloudPlatform/cloud-builders-community.git",
            ],
        },
        {
            "id": "move helm to root",
            "name": "alpine",
            "entrypoint": "sh",
            "args": [
                "-c",
                "mv cloud-builders-community/helm/* . "
                "&& rm -rf cloud-builders-community",
            ],
        },
    ]

    build_id = trigger_docker_build(
        project_name,
        service_account_file,
        package_name=PACKAGE_NAME_DOCKER_HELM,
        tag="latest",
        preprocessing_steps=steps,
        context_tar=tar_path,
    )

    wait_for_build_to_finish(
        project_name, build_id, service_account_file, PACKAGE_NAME_DOCKER_HELM
    )


def generate_password() -> Text:
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(20))


def write_helm_values(path: Text, project_name: Text) -> None:
    values = {
        "rasax": {
            "initialUser": {"password": generate_password()},
            "passwordSalt": generate_password(),
            "token": generate_password(),
            "jwtSecret": generate_password(),
            "tag": "0.29.3",
        },
        "rasa": {"token": generate_password(), "tag": "1.10.3-full"},
        "rabbitmq": {"rabbitmq": {"password": generate_password()}},
        "app": {
            "name": f"gcr.io/{project_name}/{PACKAGE_NAME_DOCKER_ACTIONS}",
            "tag": "latest",
        },
        "global": {
            "postgresql": {"postgresqlPassword": generate_password()},
            "redis": {"password": generate_password()},
        },
    }
    write_yaml_file(values, path)


def build_infrastructure(
    project_name: Text, service_account_file: Text
) -> Dict[Text, Any]:
    print_info("Preparing infrastructure for the deployment...")
    create_storage_bucket(
        project_name,
        service_account_file,
        f"{project_name}_{PACKAGE_NAME_DOCKER_TERRAFORRM}_state",
    )

    copy_tree(
        pkg_resources.resource_filename(PACKAGE_NAME, "cli/deployment"), "deployment"
    )

    terraform_main = read_file("deployment/main.tf")
    terraform_main = terraform_main.replace("${PROJECT_NAME}", project_name).replace(
        "${PACKAGE_NAME_DOCKER_TERRAFORRM}", PACKAGE_NAME_DOCKER_TERRAFORRM
    )
    write_text_file(terraform_main, "deployment/main.tf")

    terraform_vars = {
        "project": project_name,
        "name": "rasa",  # TODO: change
        "region": "europe-west1",
    }
    create_terraform_vars("deployment", terraform_vars)

    tar_path = create_temporary_tar("deployment")

    steps = [
        {
            "id": "show directory",
            "name": "alpine",
            "entrypoint": "sh",
            "args": ["-c", "ls -lisa deployment"],
        },
        {
            "id": "terraform init",
            "name": "hashicorp/terraform:0.12.28",
            "entrypoint": "sh",
            "args": ["-c", "cd deployment && terraform init"],
        },
        {
            "id": "terraform plan",
            "name": "hashicorp/terraform:0.12.28",
            "entrypoint": "sh",
            "args": ["-c", "cd deployment && terraform plan"],
        },
        {
            "id": "terraform apply",
            "name": "hashicorp/terraform:0.12.28",
            "entrypoint": "sh",
            "args": ["-c", "cd deployment && terraform apply -auto-approve"],
        },
    ]

    build_id = trigger_cloud_build(
        project_name,
        service_account_file,
        package_name=PACKAGE_NAME_DOCKER_TERRAFORRM,
        context_tar=tar_path,
        steps=steps,
    )

    wait_for_build_to_finish(
        project_name, build_id, service_account_file, PACKAGE_NAME_DOCKER_TERRAFORRM
    )

    return terraform_vars


def deploy_application(
    project_name: Text,
    service_account_file: Text,
    compute_region: Text,
    container_cluster: Text,
) -> Any:
    print_info("Deploying your assistant in the cloud...")
    helm_values_path = os.path.join("deployment", "values.yml")
    if not os.path.exists(helm_values_path):
        write_helm_values(helm_values_path, project_name)
    helm_values = read_yaml_file(helm_values_path)

    tar_path = create_temporary_tar(helm_values_path, arcname="values.yml")

    kube_env = [
        f"CLOUDSDK_COMPUTE_REGION={compute_region}",
        f"CLOUDSDK_CONTAINER_CLUSTER={container_cluster}",
        f"KUBECONFIG=/workspace/.kube/config",
    ]

    steps = [
        {
            "id": "show directory",
            "name": "alpine",
            "entrypoint": "sh",
            "args": ["-c", "ls -lisa ."],
        },
        {
            "id": "configure kube workspace",
            "name": "gcr.io/cloud-builders/kubectl",
            "args": ["cluster-info"],
            "env": kube_env,
        },
        {
            "id": "helm add rasa x chart",
            "name": f"gcr.io/{project_name}/{PACKAGE_NAME_DOCKER_HELM}:latest",
            "env": ["KUBECONFIG=/workspace/.kube/config"],
            "args": ["repo", "add", "rasa-x", "https://rasahq.github.io/rasa-x-helm",],
        },
        {
            "id": "helm upgrade rasa x chart",
            "name": f"gcr.io/{project_name}/{PACKAGE_NAME_DOCKER_HELM}:latest",
            "env": ["KUBECONFIG=/workspace/.kube/config"],
            "args": [
                "upgrade",
                "--install",
                "rasa-bot",
                "rasa-x/rasa-x",
                # "--reuse-values",
                "--values",
                "values.yml",
                "--set",
                "nginx.service.type=LoadBalancer",
            ],
        },
        {
            "id": "wait for deployment",
            "name": "gcr.io/cloud-builders/kubectl",
            "args": [
                "wait",
                "--for=condition=available",
                "--timeout=20m",
                "--selector",
                "app.kubernetes.io/component=rasa-x",
                "deployment",
            ],
            "env": kube_env,
        },
        {
            "id": "external-ip",
            "name": "gcr.io/cloud-builders/kubectl",
            "entrypoint": "sh",
            "args": [
                "-c",
                "/builder/kubectl.bash version --short && "
                "kubectl get svc -l 'app.kubernetes.io/component=nginx'"
                " -o=jsonpath='{.items[].status.loadBalancer.ingress[].ip}' "
                "> $$BUILDER_OUTPUT/output",
            ],
            "env": kube_env,
        },
    ]

    build_id = trigger_cloud_build(
        project_name,
        service_account_file,
        package_name=PACKAGE_NAME_DOCKER_RASA,
        context_tar=tar_path,
        steps=steps,
    )

    wait_for_build_to_finish(
        project_name, build_id, service_account_file, PACKAGE_NAME_DOCKER_RASA
    )
    response = check_build(project_name, build_id, service_account_file)
    ip_step_output = response.get("results", {}).get("buildStepOutputs", [""])[-1]
    ingress_ip = base64.b64decode(ip_step_output).decode("utf-8")
    helm_values["ip"] = ingress_ip
    return helm_values


def move_file_from_package_to_project(
    package_file_path: Text, target_filename: Text
) -> None:
    template_file = pkg_resources.resource_filename(PACKAGE_NAME, package_file_path)
    with open(template_file) as f:
        with open(target_filename, mode="w") as target:
            target.write(f.read())


def create_dockerfile(path: Text) -> None:
    if os.path.exists(path):
        print_info("Found a 'Dockerfile', assuming it is the action server...")
        return

    print_info("Creating a default Dockerfile for your action server...")
    move_file_from_package_to_project(DOCKERFILE_TEMPLATE, path)

    print_success("Successfully created a Dockerfile for your action server.")


def update_role_binding(
    policy: Dict[Text, Any], role: Text, service_account_name: Text
):
    for binding in policy.get("bindings", []):
        if binding.get("role") == role:
            if service_account_name not in binding.get("members", []):
                binding["members"].append(service_account_name)
            break
    else:
        policy["bindings"].append(
            {"role": role, "members": [service_account_name],}
        )


def set_cloud_build_permissions(project_name: Text, service_account_file: Text) -> None:
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )
    resource_manager_client = discovery.build(
        "cloudresourcemanager", "v1", credentials=credentials, cache_discovery=False
    )

    response = resource_manager_client.projects().get(projectId=project_name).execute()
    project_number = response.get("projectNumber")

    if not project_number:
        print_error("Failed to get project number :( Response:")
        logger.error(response)
        sys.exit(1)

    cloudbuild_sa = f"serviceAccount:{project_number}@cloudbuild.gserviceaccount.com"

    policy = (
        resource_manager_client.projects().getIamPolicy(resource=project_name).execute()
    )

    original_policy = copy.deepcopy(policy)

    update_role_binding(policy, "roles/editor", cloudbuild_sa)
    update_role_binding(policy, "roles/resourcemanager.projectIamAdmin", cloudbuild_sa)
    update_role_binding(policy, "roles/container.admin", cloudbuild_sa)

    if original_policy != policy:
        logger.debug("Need to update IAM policy for cloubuild account...")
        set_iam_policy_request_body = {"policy": policy}

        request = resource_manager_client.projects().setIamPolicy(
            resource=project_name, body=set_iam_policy_request_body
        )
        request.execute()
        print_success(f"Updated IAM policy for {cloudbuild_sa}")


def get_available_models(base_url: Text, auth_headers: Dict[Text, Any]) -> Set[Text]:
    r = requests.get(f"{base_url}/api/projects/default/models", headers=auth_headers)

    if r.status_code != 200:
        logger.error(
            f"Failed to fetch available models from instance "
            f"(status {r.status_code}), assuming there are none..."
        )
        logger.error(r.text)
        return set()
    else:
        return {m["model"] for m in r.json()}


def wait_till_rasa_is_alive(base_url: Text) -> None:
    with tqdm(desc="Waiting for Rasa Open Source to be ready") as pbar:
        while True:
            response = requests.get(f"{base_url}/api/health")
            if response.json().get("production", {}).get("status") == 200:
                return
            else:
                time.sleep(1)
                pbar.refresh()


def upload_model_to_server(
    model_name: Text, model_path: Text, base_url: Text, auth_headers: Dict[Text, Any]
) -> bool:
    with open(model_path, "rb") as f:
        files = {"model": f}

        r = requests.post(
            f"{base_url}/api/projects/default/models", files=files, headers=auth_headers
        )

        if r.status_code == 409:
            logger.info(
                "Most recent model already exists on server, continuing without upload."
            )
            return False
        elif r.status_code >= 300:
            logger.error(
                f"Error while trying to upload model to your instance "
                f"(status {r.status_code}), skipping model upload..."
            )
            logger.error(r.text)
            return False

    with tqdm(desc="Waiting for model to be ready") as pbar:
        while True:
            available_models = get_available_models(base_url, auth_headers)
            if model_name in available_models:
                break
            else:
                pbar.refresh()
                time.sleep(5)
    return True


def upload_model(
    deployment_address: Text, username: Text, password: Text, model_path: Text
) -> None:
    if not model_path:
        logger.info("No trained model present. Skipping model upload.")
        return

    print_info("Uploading ML model...")
    base_url = f"http://{deployment_address}:8000"
    response = requests.post(
        f"{base_url}/api/auth", json={"username": username, "password": password}
    )
    response.raise_for_status()
    token = response.json().get("access_token", "")

    auth_headers = {"Authorization": f"Bearer {token}"}

    wait_till_rasa_is_alive(base_url)

    available_models = get_available_models(base_url, auth_headers)
    model_name = Path(model_path).stem[
        :-4
    ]  # removes '.tar', '.gz' gets removed by stem

    if model_name not in available_models:
        model_was_uploaded = upload_model_to_server(
            model_name, model_path, base_url, auth_headers
        )

        if model_was_uploaded:
            r = requests.put(
                f"{base_url}/api/projects/default/models/{model_name}/tags/production",
                headers=auth_headers,
            )
            r.raise_for_status()


def get_cloud_provider(args: argparse.Namespace) -> Text:
    if args.provider:
        return args.provider
    else:
        return ask_cloud_provider()


def ask_cloud_provider() -> Text:
    q = questionary.select(
        "Which cloud do you want to deploy to?",
        choices=[
            Choice("Google Cloud Cluster (recommended)", value="gcloud"),
            Choice(
                "Google Cloud Single Machine (not implemented)", value="gcloud-bare"
            ),
            Choice("Amazon Web Services (not implemented)", value="aws"),
        ],
    )
    return q.ask()


def print_command_info(
    args: argparse.Namespace, project_name: Text, service_account_file: Text
) -> None:
    if args.project_id or args.service_account or args.provider:
        # user probably already knows about the arguments
        return

    print(
        f"Note: if you want to avoid filling these questions for future deploys,"
        f"you can use flags instead:\n"
        f"rasa deploy --provider gcloud --project-id {project_name} "
        f"--service-account {service_account_file}"
    )


def print_login_info(ip: Text, username: Text, password: Text) -> None:
    print_success(
        f"Your instance was successfully deployed to google cloud. "
        f"You can access your instance at: \n"
        f"http://{ip}:8000/login?username={username}&password={password}"
    )


def deploy_on_gcloud(args: argparse.Namespace) -> None:
    project_name = get_project_name(args)
    service_account_file = get_service_account(args, project_name)

    print_command_info(args, project_name, service_account_file)
    logger.info("Let me check if all of that works...")

    enable_gcloud_services(
        project_name, service_account_file, REQUIRED_GOOGLE_CLOUD_SERVICES,
    )

    set_cloud_build_permissions(project_name, service_account_file)
    build_actions_docker(project_name, service_account_file)
    build_helm_docker(project_name, service_account_file)

    meta = build_infrastructure(project_name, service_account_file)
    helm_values = deploy_application(
        project_name, service_account_file, meta["region"], meta["name"]
    )

    user_password = helm_values["rasax"]["initialUser"]["password"]
    upload_model(
        helm_values["ip"], "me", user_password, get_latest_model(),
    )
    print_login_info(helm_values["ip"], "me", user_password)


def configure_deployment(args: argparse.Namespace) -> None:
    print_success("Let's get your bot deployed in the cloud 📦\n")

    cloud_name = get_cloud_provider(args)

    if cloud_name == "gcloud":
        deploy_on_gcloud(args)

    elif cloud_name is not None:
        raise NotImplementedError(
            f"Cloud provider {cloud_name} has not been implemented yet."
        )
