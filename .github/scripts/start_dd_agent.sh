#!/bin/bash

DD_API_KEY=$1
USE_GPU=$2
dataset=$3
config=$4

echo "Dataset: ${dataset}"
echo "Config: ${config}"

# Install Datadog system agent
DD_AGENT_MAJOR_VERSION=7 DD_API_KEY=$DD_API_KEY DD_SITE="datadoghq.eu" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
DATADOG_YAML_PATH=/etc/datadog-agent/datadog.yaml
sudo chmod 666 $DATADOG_YAML_PATH

# Associate metrics with tags and env
{
    echo "env: rasa-regression-tests"
    echo "tags:"
    echo "- service:rasa"
    echo "- dataset:$dataset"
    echo "- config:$config"
    echo ""
    echo "process_config:"
    echo "    enabled: false"
    echo "apm_config:"
    echo "    enabled: false"
    echo "use_dogstatsd: true"
} >> $DATADOG_YAML_PATH

set -x

# Enable system_core integration
sudo mv /etc/datadog-agent/conf.d/system_core.d/conf.yaml.example /etc/datadog-agent/conf.d/system_core.d/conf.yaml

if [[ "${USE_GPU}" == "true" ]]; then
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
