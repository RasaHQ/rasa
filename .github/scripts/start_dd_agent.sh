#!/bin/bash

DD_API_KEY=$1
ACCELERATOR_TYPE=$2

echo "ACCELERATOR_TYPE: ${ACCELERATOR_TYPE}"
echo "DATASET: ${DATASET}"
echo "CONFIG: ${CONFIG}"
echo "DATASET_COMMIT: ${DATASET_COMMIT}"
echo "BRANCH: ${BRANCH}"
echo "GIT_SHA: ${GIT_SHA}"

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
    echo "- git_sha:${GIT_SHA}"
    echo ""
    echo "process_config:"
    echo "    enabled: false"
    echo "apm_config:"
    echo "    enabled: false"
    echo "use_dogstatsd: true"
} >> $DATADOG_YAML_PATH

set -x

nvidia-smi

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

sudo ls -al /etc/datadog-agent/conf.d/nvml.d/
sudo cat /etc/datadog-agent/conf.d/nvml.d/conf.yaml
sleep 10
sudo datadog-agent status

nvidia-smi

INSTALL_DIR="/opt/datadog-agent"
AGENTPATH="$INSTALL_DIR/bin/agent/agent"
PIDFILE="$INSTALL_DIR/run/agent.pid"
AGENT_ARGS="run -p $PIDFILE"
AGENT_USER="dd-agent"
LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

sudo -E start-stop-daemon --start --background --quiet  --chuid $AGENT_USER  --pidfile $PIDFILE --user $AGENT_USER --startas /bin/bash -- -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH $AGENTPATH $AGENT_ARGS"
