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
sudo service datadog-agent restart

sudo ls -al /etc/datadog-agent/conf.d/nvml.d/
sudo cat /etc/datadog-agent/conf.d/nvml.d/conf.yaml
sleep 10
sudo datadog-agent status

nvidia-smi

which nvidia-smi
NVIDIA_SMI_PATH=$(which nvidia-smi)
ls -al "$NVIDIA_SMI_PATH"

ls -al /usr/lib
ls -al /usr/lib/nvidia-367
ls -al /usr/lib/nvidia

sudo su - dd-agent -s /bin/bash -c "$NVIDIA_SMI_PATH"
sudo su - dd-agent -s /bin/bash -c /usr/local/nvidia/bin/nvidia-smi
sudo su - dd-agent -s /bin/bash -c /usr/bin/nvidia-smi
sudo su - dd-agent -s /bin/bash -c nvidia-smi
