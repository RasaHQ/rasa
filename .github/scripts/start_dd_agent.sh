#!/bin/bash

DD_API_KEY=$1
ACCELERATOR_TYPE=$2
NVML_INTERVAL_IN_SEC=${3:-15}  # 15 seconds are the default interval

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
    echo "- dataset:${DATASET_NAME}"
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
    echo "- index_repetition:${INDEX_REPETITION}"
    echo "- host_name:${HOST_NAME}"
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
    NVML_CONF_FPATH="/etc/datadog-agent/conf.d/nvml.d/conf.yaml"
    sudo mv "${NVML_CONF_FPATH}.example" ${NVML_CONF_FPATH}
    if [[ "${NVML_INTERVAL_IN_SEC}" != 15 ]]; then
        # Append a line to the NVML config file
        sudo echo "    min_collection_interval: ${NVML_INTERVAL_IN_SEC}" | sudo tee -a ${NVML_CONF_FPATH} > /dev/null
    fi
fi

# Apply changes
sudo service datadog-agent stop

# Restart agent (such that GPU/NVML metrics are collected)
# Adusted code from /etc/init/datadog-agent.conf
INSTALL_DIR="/opt/datadog-agent"
AGENTPATH="$INSTALL_DIR/bin/agent/agent"
PIDFILE="$INSTALL_DIR/run/agent.pid"
AGENT_USER="dd-agent"
LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
sudo -E start-stop-daemon --start --background --quiet --chuid $AGENT_USER --pidfile $PIDFILE --user $AGENT_USER --startas /bin/bash -- -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH $AGENTPATH run -p $PIDFILE"

# Adusted code from /etc/init/datadog-agent-process.conf
TRACE_AGENTPATH="$INSTALL_DIR/embedded/bin/trace-agent"
TRACE_PIDFILE="$INSTALL_DIR/run/trace-agent.pid"
sudo -E start-stop-daemon --start --background --quiet --chuid $AGENT_USER --pidfile $TRACE_PIDFILE --user $AGENT_USER --startas /bin/bash -- -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH $TRACE_AGENTPATH --config $DATADOG_YAML_PATH --pid $TRACE_PIDFILE"

# Adusted code from /etc/init/datadog-agent-trace.conf
PROCESS_AGENTPATH="$INSTALL_DIR/embedded/bin/process-agent"
PROCESS_PIDFILE="$INSTALL_DIR/run/process-agent.pid"
SYSTEM_PROBE_YAML="/etc/datadog-agent/system-probe.yaml"
sudo -E start-stop-daemon --start --background --quiet --chuid $AGENT_USER --pidfile $PROCESS_PIDFILE --user $AGENT_USER --startas /bin/bash -- -c "LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PROCESS_AGENTPATH --config=$DATADOG_YAML_PATH --sysprobe-config=$SYSTEM_PROBE_YAML --pid=$PROCESS_PIDFILE"
