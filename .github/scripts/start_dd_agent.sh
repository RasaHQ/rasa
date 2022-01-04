#!/bin/bash

DD_API_KEY=$1
ACCELERATOR_TYPE=$2

# Install Datadog system agent
DD_AGENT_MAJOR_VERSION=7 DD_API_KEY=$DD_API_KEY DD_SITE="datadoghq.eu" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
DATADOG_YAML_PATH=/etc/datadog-agent/datadog.yaml
sudo chmod 666 $DATADOG_YAML_PATH

# Associate metrics with tags and env
{
    echo "env: rasa-regression-tests"
    echo "tags:"
    echo "- service:rasa"
    echo "- accelerator_type:${ACCELERATOR_TYPE}"
    echo "- dataset:${DATASET}"
    echo "- config:${CONFIG}"
    echo "- dataset_commit:${DATASET_COMMIT}"
    echo "- branch:${BRANCH}"
    echo "- github_sha:${GITHUB_SHA}"
    echo "- pr_id:${PR_ID:-schedule}"
    echo "- pr_url:${PR_URL:-schedule}"
    echo "- type:${TYPE}"
    echo "- dataset_repository_branch:${DATASET_REPOSITORY_BRANCH}"
    echo "- external_dataset_repository:${IS_EXTERNAL:-none}"
    echo "- config_repository:training-data"
    echo "- config_repository_branch:${DATASET_REPOSITORY_BRANCH}"
    echo "- workflow:${GITHUB_WORKFLOW:-none}"
    echo "- github_run_id:${GITHUB_RUN_ID:-none}"
    echo "- github_event:${GITHUB_EVENT_NAME:-none}"
    echo ""
    echo "apm_config:"
    echo "    enabled: true"
    echo "process_config:"
    echo "    enabled: false"
    echo "use_dogstatsd: true"
} >> $DATADOG_YAML_PATH

# Enable system_core integration
sudo mv /etc/datadog-agent/conf.d/system_core.d/conf.yaml.example /etc/datadog-agent/conf.d/system_core.d/conf.yaml

if [[ "${ACCELERATOR_TYPE}" == "GPU" ]]; then
# Install and enable NVML integration
sudo datadog-agent integration --allow-root install -t datadog-nvml==1.0.1
sudo -u dd-agent -H /opt/datadog-agent/embedded/bin/pip3 install grpcio pynvml
sudo mv /etc/datadog-agent/conf.d/nvml.d/conf.yaml.example /etc/datadog-agent/conf.d/nvml.d/conf.yaml
fi

# Apply changes
sudo service datadog-agent stop
sudo service datadog-agent restart

sleep 10
sudo datadog-agent status
