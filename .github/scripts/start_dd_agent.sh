#!/bin/bash

set -x

sudo netstat -plnt

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
    echo "process_config:"
    echo "    enabled: false"
    echo "apm_config:"
    echo "    enabled: true"
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

INSTALL_DIR="/opt/datadog-agent"
AGENTPATH="$INSTALL_DIR/bin/agent/agent"
PIDFILE="$INSTALL_DIR/run/agent.pid"
AGENT_ARGS="run -p $PIDFILE"
AGENT_USER="dd-agent"
LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

sudo -E start-stop-daemon --start --background --quiet --chuid $AGENT_USER --pidfile $PIDFILE --user $AGENT_USER --startas /bin/bash -- -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH $AGENTPATH $AGENT_ARGS"

sudo netstat -plnt

sudo datadog-agent status

sleep 10

sudo datadog-agent status
