## Makefile for Rasa Open Source
## This Makefile provides a set of commands to run tests, linting, formatting, and other development tasks.
## It is intended to be used by developers working on Rasa and Rasa CI.
## When adding a new command, please make sure to add a description for it.
## You can run `make help` to see all available commands.
.PHONY: clean test lint init docs format formatter build-docker

# If makefile_vars is not included, include it.
# It is a good practice to check if common variables are included across all makefiles.
ifndef MAKEFILE_VAR
include ./makefile_vars
endif

ifndef DOCKER_BUILD_COMMAND
include ./Makefile.docker
endif

JOBS ?= 1
INTEGRATION_TEST_FOLDER = tests/integration_tests
INTEGRATION_TEST_PYTEST_MARKERS ?= "sequential or broker or concurrent_lock_store or ((not sequential) and (not broker) and (not concurrent_lock_store))"
TRACING_INTEGRATION_TEST_FOLDER = $(INTEGRATION_TEST_FOLDER)/tracing
METRICS_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/tracing/test_metrics.py
CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/actions/custom_actions
NLU_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)/test_custom_actions_with_nlu.py
CALM_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)/test_custom_actions_with_calm.py
ENTERPRISE_SEARCH_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/enterprise_search
CHANNEL_CONNECTOR_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/channels
VOICE_READY_CONNECTOR_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/channels/voice_ready
VOICE_STREAM_CONNECTOR_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/channels/voice_stream
TRACKER_STORE_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/tracker_stores
CUSTOM_COMPONENT_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/custom_components
CALM_PII_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/privacy
INTEGRATION_TEST_DEPLOYMENT_PATH = $(PWD)/tests_deployment
TRANSFORMERS_OFFLINE ?= 1

BOT_PATH ?=
MODEL_NAME ?= model

# find user's id
USER_ID := $(shell id -u)


help:  ## show help message
	@sed \
		-e '/^[a-zA-Z0-9_\-]*:.*##/!d' \
		-e 's/:.*##\s*/:/' \
		-e 's/^\(.\+\):\(.*\)/$(shell tput setaf 6)\1$(shell tput sgr0):\2/' \
		$(MAKEFILE_LIST) | column -c2 -t -s :
#	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean:  ## Remove Python/build artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf dist/

install:  ## Install rasa and it's dependencies without extras.
	poetry run python -m pip install -U pip
	poetry install

install-mitie:  ## Install mitie.
	poetry run python -m pip install -U pip
	poetry run python -m pip install -U git+https://github.com/tmbo/MITIE.git#egg=mitie

install-full: install-mitie  ## Install rasa with all extras (transformers, tensorflow_text, spacy, jieba).
	poetry install -E full

install-pii:  ## Install rasa-pro with PII optional dependencies.
	poetry run python -m pip install -U pip
	poetry install -E pii

format: ## Apply ruff formatting to code.
	poetry run ruff format rasa tests
	poetry run ruff check rasa tests --ignore D --fix

lint:  ## Lint code with ruff, and check if ruff formatter should be applied.
     # Ignore docstring errors when running on the entire project
	poetry run ruff check rasa tests --ignore D
	poetry run ruff format --check rasa tests
	make lint-docstrings

# Compare against `main` if no branch was provided
BRANCH ?= main
lint-docstrings:  ## Check docstring conventions in changed files.
	./scripts/lint_python_docstrings.sh $(BRANCH)

lint-changelog:  ## Check if changelog files are up to date.
	./scripts/lint_changelog_files.sh

lint-security:  ## Check for security issues using bandit.
	poetry run bandit -ll -ii -r --config pyproject.toml rasa/*

types:  ## Check for type errors using mypy.
	poetry run mypy rasa

static-checks: lint lint-security types  ## Run all python static checks.

update-rasa-sdk-to-main: ## Update the Rasa SDK to the latest version on the main branch.
	poetry run python -m pip install git+https://github.com/RasaHQ/rasa-sdk.git@main

prepare-spacy:  ## Download models needed for spacy tests.
	poetry run python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
	poetry run python -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl

prepare-mitie:  ## Download the standard english MITIE model.
	wget --progress=dot:giga -N -P data/ https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2
ifeq ($(OS),Windows_NT)
	7z x data/MITIE-models-v0.2.tar.bz2 -bb3
	7z x MITIE-models-v0.2.tar -bb3
	cp MITIE-models/english/total_word_feature_extractor.dat data/
	rm -r MITIE-models
	rm MITIE-models-v0.2.tar
else
	tar -xvjf data/MITIE-models-v0.2.tar.bz2 --strip-components 2 -C data/ MITIE-models/english/total_word_feature_extractor.dat
endif
	rm data/MITIE*.bz2

prepare-transformers: ## Download all models needed for testing transformers.
	while read -r MODEL; \
	do poetry run python scripts/download_transformer_model.py $$MODEL ; \
	done < data/test/hf_transformers_models.txt
	if ! [ $(CI) ]; then \
  		poetry run python scripts/download_transformer_model.py rasa/LaBSE; \
  	fi
prepare-tests-macos:  ## Install system requirements for running tests on macOS.
	brew install wget graphviz || true

prepare-tests-ubuntu: ## Install system requirements for running tests on Ubuntu and Debian based systems.
	sudo apt-get -y install graphviz graphviz-dev python3-tk

prepare-tests-windows:  ## Install system requirements for running tests on Windows.
	choco install wget graphviz

# GitHub Action has pre-installed a helper function for installing Chocolatey packages
# It will retry the installation 5 times if it fails
# See: https://github.com/actions/virtual-environments/blob/main/images/win/scripts/ImageHelpers/ChocoHelpers.ps1
prepare-tests-windows-gha:
	powershell -command "Install-ChocoPackage wget graphviz"

test: clean  ## Run Rasa unit tests using pytest.
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest tests \
			-n $(JOBS) \
			--dist loadscope \
			--cov rasa \
			--ignore $(INTEGRATION_TEST_FOLDER)/

test-integration:  ## Run general integration tests using pytest. It will run all integration tests except ones for metrics, tracing, custom actions and enterprise search.
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
ifeq (,$(wildcard $(INTEGRATION_TEST_DEPLOYMENT_PATH)/.env))
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest $(INTEGRATION_TEST_FOLDER)/ \
			-n $(JOBS) \
			-m $(INTEGRATION_TEST_PYTEST_MARKERS) \
			--dist loadgroup  \
			--ignore $(TRACING_INTEGRATION_TEST_FOLDER) \
			--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
			--ignore $(ENTERPRISE_SEARCH_INTEGRATION_TEST_PATH) \
			--ignore $(TRACKER_STORE_INTEGRATION_TEST_PATH) \
			--ignore $(CHANNEL_CONNECTOR_INTEGRATION_TEST_PATH) \
			--ignore $(CUSTOM_COMPONENT_INTEGRATION_TEST_PATH) \
			--ignore $(CALM_PII_INTEGRATION_TEST_PATH) \
			--junitxml=report_integration.xml
else
	set -o allexport; \
	source $(INTEGRATION_TEST_DEPLOYMENT_PATH)/.env && \
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest $(INTEGRATION_TEST_FOLDER)/ \
			-n $(JOBS) \
			-m $(INTEGRATION_TEST_PYTEST_MARKERS) \
			--dist loadgroup \
			--ignore $(TRACING_INTEGRATION_TEST_FOLDER) \
			--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
			--ignore $(ENTERPRISE_SEARCH_INTEGRATION_TEST_PATH) \
			--ignore $(TRACKER_STORE_INTEGRATION_TEST_PATH) \
			--ignore $(CHANNEL_CONNECTOR_INTEGRATION_TEST_PATH) \
			--junitxml=report_integration.xml && \
	set +o allexport
endif

test-anonymization: PYTEST_MARKER=category_anonymization and (not flaky) and (not acceptance) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-anonymization: DD_ARGS := $(or $(DD_ARGS),)
test-anonymization: test-marker  ## Run anonymization tests

test-with-large-data: ## Run tests on large data set
	poetry run \
		pytest tests/acceptance_tests/large_data_tests/test_training_time.py \
			-n $(JOBS)

test-cli: PYTEST_MARKER=category_cli and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-cli: DD_ARGS := $(or $(DD_ARGS),)
test-cli: test-marker  ## Run cli tests

test-policies: PYTEST_MARKER=category_policies and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-policies: DD_ARGS := $(or $(DD_ARGS),)
test-policies: test-marker  ## Run policies tests

test-nlu-featurizers: PYTEST_MARKER=category_nlu_featurizers and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-nlu-featurizers: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-featurizers: prepare-spacy prepare-mitie prepare-transformers test-marker  ## Run nlu featurizers tests

test-nlu-predictors: PYTEST_MARKER=category_nlu_predictors and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-nlu-predictors: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-predictors: prepare-spacy prepare-mitie test-marker  ## Run nlu predictors tests

test-full-model-training: PYTEST_MARKER=category_full_model_training and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-full-model-training: DD_ARGS := $(or $(DD_ARGS),)
test-full-model-training: prepare-spacy prepare-mitie prepare-transformers test-marker  ## Run full model training tests

test-other-unit-tests: PYTEST_MARKER=category_other_unit_tests and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-other-unit-tests: DD_ARGS := $(or $(DD_ARGS),)
test-other-unit-tests: prepare-spacy prepare-mitie test-marker  ## Run other unit tests

test-performance: PYTEST_MARKER=category_performance and (not flaky) and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-performance: DD_ARGS := $(or $(DD_ARGS),)
test-performance: test-marker  ## Run performance tests

test-flaky: PYTEST_MARKER=flaky and (not acceptance) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-flaky: DD_ARGS := $(or $(DD_ARGS),)
test-flaky: prepare-spacy prepare-mitie test-marker  ## Run flaky tests

test-acceptance: PYTEST_MARKER=acceptance and (not flaky) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-acceptance: DD_ARGS := $(or $(DD_ARGS),)
test-acceptance: prepare-spacy prepare-mitie test-marker ## Run acceptance tests

test-audio-manual: PYTEST_MARKER=category_audio_manual and (not flaky) and (not category_anonymization) and (not category_large_data_tests) and (not category_dm1_tensorflow)
test-audio-manual: DD_ARGS := $(or $(DD_ARGS),)
test-audio-manual: test-marker

test-voice-integration: ## Run voice integration tests
	poetry run \
        pytest $(VOICE_READY_CONNECTOR_INTEGRATION_TEST_PATH) \
        $(VOICE_STREAM_CONNECTOR_INTEGRATION_TEST_PATH) \
        -n $(JOBS) \
        --junitxml=integration-test-results.xml

test-dm1-tensorflow: PYTEST_MARKER=category_dm1_tensorflow
test-dm1-tensorflow: prepare-spacy prepare-mitie prepare-transformers test-marker

test-gh-actions:  ## Run all tests for GitHub Actions
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest .github/tests --cov .github/scripts

test-marker: clean ## Run marker tests
    # OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
	TRANSFORMERS_OFFLINE=$(TRANSFORMERS_OFFLINE) \
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest tests \
			-n $(JOBS) \
			--dist loadscope \
			-m "$(PYTEST_MARKER)" \
			--cov=rasa \
			--cov-report=xml \
			--cov-branch \
			--ignore $(INTEGRATION_TEST_FOLDER)/ $(DD_ARGS)

release:  ## Prepare a release.
	poetry run python scripts/release.py prepare --interactive

build-tests-deployment-env: ## Create environment files (.env) for docker-compose.
	cd $(INTEGRATION_TEST_DEPLOYMENT_PATH) && \
	test -f .env || cat .env.example >> .env

run-integration-containers: build-tests-deployment-env ## Run the integration test containers.
	cd $(INTEGRATION_TEST_DEPLOYMENT_PATH) && \
	docker compose -f docker-compose.integration.yml up &

stop-integration-containers: ## Stop the integration test containers.
	cd $(INTEGRATION_TEST_DEPLOYMENT_PATH) && \
	docker compose -f docker-compose.integration.yml down

tag-release-auto:  ## Tag a release automatically.
	poetry run python scripts/release.py tag --skip-confirmation

TRACING_INTEGRATION_TEST_DEPLOYMENT_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_tracing_deployment
TRACING_INTEGRATION_TEST_DOCKER_COMPOSE = $(TRACING_INTEGRATION_TEST_DEPLOYMENT_PATH)/docker-compose.yml

train-nlu:  ## Train the simple NLU bot for tracing integration tests.
	docker run \
		--rm \
		-u $(USER_ID) \
		--name rasa-pro-training-$(RASA_IMAGE_TAG) \
		-e RASA_PRO_LICENSE=$(RASA_PRO_LICENSE) \
		-v $(TRACING_INTEGRATION_TEST_DEPLOYMENT_PATH)/simple_bot\:/app \
		$(RASA_REPOSITORY)\:$(RASA_IMAGE_TAG) \
		train --fixed-model-name model

run-tracing-integration-containers: train-nlu ## Run the tracing integration test containers.
	USER_ID=$(USER_ID) \
	docker compose \
		-f $(TRACING_INTEGRATION_TEST_DOCKER_COMPOSE) \
		up --wait

stop-tracing-integration-containers: ## Stop the tracing integration test containers.
	docker compose \
		-f $(TRACING_INTEGRATION_TEST_DOCKER_COMPOSE) \
		down

test-tracing-integration:  ## Run the tracing integration tests. Make sure to run run-tracing-integration-containers before running this target.
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	PYTHONPATH=./vendor/jaeger-python-proto \
	poetry run \
		pytest $(TRACING_INTEGRATION_TEST_FOLDER) \
			-n $(JOBS) \
			--ignore $(METRICS_INTEGRATION_TEST_PATH) \
			--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
			--junitxml=integration-results-tracing.xml


METRICS_SETUP_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_metrics_setup

# We need to set the user ID used to run rasa for Linux OS (in Github workflows user ID is 1000)
train-metrics-calm-bot:  ## Train the calm bot for tracing integration tests.
	docker run \
		--rm \
		-u $(USER_ID) \
		--name rasa-pro-training-$(RASA_IMAGE_TAG) \
		-e RASA_PRO_LICENSE=$(RASA_PRO_LICENSE) \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-v $(METRICS_SETUP_PATH)/calm_bot\:/app \
		$(RASA_REPOSITORY)\:$(RASA_IMAGE_TAG) \
		train --fixed-model-name model

run-metrics-integration-containers: train-metrics-calm-bot ## Run the metrics integration test containers.
	USER_ID=$(USER_ID) \
	docker compose \
		-f $(METRICS_SETUP_PATH)/docker-compose.yml \
		up --wait

stop-metrics-integration-containers: ## Stop the metrics integration test containers.
	docker compose \
		-f $(METRICS_SETUP_PATH)/docker-compose.yml \
		down

test-metrics-integration:  ## Run the metrics integration tests. Make sure to run run-metrics-integration-containers before running this target.
	poetry run \
		pytest $(METRICS_INTEGRATION_TEST_PATH) \
			-n $(JOBS) \
			--junitxml=integration-results-metric.xml


ACTION_SERVER_INTEGRATION_TESTS_DEPLOYMENT_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_custom_action_server
ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH = $(ACTION_SERVER_INTEGRATION_TESTS_DEPLOYMENT_PATH)/docker-compose.yml
ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE = $(ACTION_SERVER_INTEGRATION_TESTS_DEPLOYMENT_PATH)/env-file
NLU_BOT_DIRECTORY = simple_nlu_bot
CALM_BOT_DIRECTORY = simple_calm_bot

# Command to train the bot.
# This command is used by both NLU and CALM bot training targets.
# Reason for not using a target is that we need to pass the
# DOCKER_ENV_VARS, CONTAINER_NAME, BOT_PATH and MODEL_NAME variables
# in two different targets and if in the future we want to run both NLU and CALM tests in one command,
# we cannot reuse the target with different variables.
TRAIN_BOT_COMMAND = docker run --rm \
		-u $(USER_ID) \
		--name $(CONTAINER_NAME) \
		$(DOCKER_ENV_VARS) \
		-v $(BOT_PATH)\:/app \
		$(RASA_REPOSITORY):$(RASA_IMAGE_TAG) \
		train --fixed-model-name $(MODEL_NAME)

train-action-server-nlu-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=$(RASA_PRO_LICENSE)
train-action-server-nlu-bot: CONTAINER_NAME = rasa-pro-training-nlu-bot-$(RASA_IMAGE_TAG)
train-action-server-nlu-bot: BOT_PATH = $(ACTION_SERVER_INTEGRATION_TESTS_DEPLOYMENT_PATH)/$(NLU_BOT_DIRECTORY)
train-action-server-nlu-bot: ## Train the NLU bot for action server integration tests.
	$(TRAIN_BOT_COMMAND)



train-action-server-calm-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=$(RASA_PRO_LICENSE) -e OPENAI_API_KEY=$(OPENAI_API_KEY)
train-action-server-calm-bot: CONTAINER_NAME = rasa-pro-training-calm-bot-$(RASA_IMAGE_TAG)
train-action-server-calm-bot: BOT_PATH = $(ACTION_SERVER_INTEGRATION_TESTS_DEPLOYMENT_PATH)/$(CALM_BOT_DIRECTORY)
train-action-server-calm-bot: ## Train the CALM bot for action server integration tests.
	$(TRAIN_BOT_COMMAND)

# Command to run the action server containers.
# This command is used by both NLU and CALM action server targets.
# Reason for not using a target is that we need to pass the BOT_PATH variable in two different targets
# and if in the future we want to run both NLU and CALM containers in one command,
# we cannot reuse the target with different variables.
# Note: ATM we are reusing same docker compose to run both NLU and CALM containers.
# Inside the docker-compose are exposing Rasa container on ports 5010, 5011,
# 5012 and 5013 (which are also tied in the tests test_custom_actions_with_nlu.py test_custom_actions_with_calm.py)
# and we cannot run both NLU and CALM action server containers at the same time.
RUN_ACTION_SERVER_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE) \
		up --wait

# This target is mutually exclusive with run-action-server-calm-containers
run-action-server-nlu-containers: BOT_PATH=./$(NLU_BOT_DIRECTORY) ## Run the action server integration test containers.
run-action-server-nlu-containers: ## Run the action server integration test containers for NLU bot.
	$(RUN_ACTION_SERVER_CONTAINERS_COMMAND)


# This target is mutually exclusive with run-action-server-nlu-containers
run-action-server-calm-containers: BOT_PATH = ./$(CALM_BOT_DIRECTORY) ## Run the action server integration test containers.
run-action-server-calm-containers: ## Run the action server integration test containers for CALM bot.
	$(RUN_ACTION_SERVER_CONTAINERS_COMMAND)


# Command to stop the action server containers.
# This command is used by both NLU and CALM to stop action server targets.
# Reason for not using a target is that we need to pass the BOT_PATH variable in two different targets
# and if in the future we want to stop both NLU and CALM action server containers in one command,
# we cannot reuse the target with different variables.
STOP_ACTION_SERVER_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE) \
		down

stop-action-server-nlu-containers: BOT_PATH = "./$(NLU_BOT_DIRECTORY)"
stop-action-server-nlu-containers: ## Stop the action server integration test containers for NLU bot.
	$(STOP_ACTION_SERVER_CONTAINERS_COMMAND)

stop-action-server-calm-containers: BOT_PATH = "./$(CALM_BOT_DIRECTORY)"
stop-action-server-calm-containers: ## Stop the action server integration test containers for CALM bot.
	$(STOP_ACTION_SERVER_CONTAINERS_COMMAND)


# Command to run the action server integration tests.
# This command is used by both the action server integration test targets.
# Reason for not using a target is that we need to pass the INTEGRATION_TEST_PATH and RESULTS_FILE variables
# in two different targets and if in the future we want to run both NLU and CALM tests in one command,
# we cannot reuse the target with different variables.
TEST_CUSTOM_ACTION_INTEGRATION_COMMAND = poetry run \
		pytest $(INTEGRATION_TEST_PATH) \
		-n $(JOBS) \
		--junitxml=$(RESULTS_FILE)


# Run the action server integration tests with NLU bot
test-custom-action-integration-with-nlu-bot: INTEGRATION_TEST_PATH = $(NLU_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)
test-custom-action-integration-with-nlu-bot: RESULTS_FILE = integration-results-custom-actions-with-nlu-bot-results.xml
test-custom-action-integration-with-nlu-bot:  ## Run the action server integration tests with NLU bot.
	$(TEST_CUSTOM_ACTION_INTEGRATION_COMMAND)


# Run the action server integration tests with CALM bot
test-custom-action-integration-with-calm-bot: INTEGRATION_TEST_PATH = $(CALM_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)
test-custom-action-integration-with-calm-bot: RESULTS_FILE = integration-results-custom-actions-with-calm-bot-results.xml
test-custom-action-integration-with-calm-bot:  ## Run the action server integration tests with CALM bot.
	$(TEST_CUSTOM_ACTION_INTEGRATION_COMMAND)

RASA_CALM_DEMO_BOT_SETUP_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_enterprise_search

RUN_RASA_CALM_DEMO_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(RASA_CALM_DEMO_BOT_SETUP_PATH)/$(DOCKER_COMPOSE_FILE) \
		up --wait

TRAIN_ENTERPRISE_SEARCH_BOT_COMMAND = docker run --rm \
		-u $(USER_ID) \
		--name $(CONTAINER_NAME) \
		$(DOCKER_ENV_VARS) \
		-v $(BOT_PATH)\:/app \
		-v $(RASA_CALM_DEMO_BOT_SETUP_PATH)/configs/${CONFIG_NAME}:/app/config.yml \
		-v $(RASA_CALM_DEMO_BOT_SETUP_PATH)/configs/${ENDPOINTS_NAME}:/app/endpoints.yml \
		${RASA_REPOSITORY}:${RASA_IMAGE_TAG} \
		train --fixed-model-name $(MODEL_NAME)

STOP_RASA_CALM_DEMO_CONTAINERS = docker compose \
		-f $(RASA_CALM_DEMO_BOT_SETUP_PATH)/${DOCKER_COMPOSE_FILE} \
		down

train-rasa-calm-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=${RASA_PRO_LICENSE} -e OPENAI_API_KEY=${OPENAI_API_KEY}
train-rasa-calm-bot: CONTAINER_NAME = rasa-calm-demo-bot-${RASA_IMAGE_TAG}
train-rasa-calm-bot: BOT_PATH = $(RASA_CALM_DEMO_BOT_SETUP_PATH)/rasa-calm-demo-bot
train-rasa-calm-bot: CONFIG_NAME = ${CONFIG}
train-rasa-calm-bot: ENDPOINTS_NAME = ${ENDPOINTS}
train-rasa-calm-bot: ## Train the CALM bot for action server integration tests.
	$(TRAIN_ENTERPRISE_SEARCH_BOT_COMMAND)

run-rasa-calm-demo-test-containers: DOCKER_COMPOSE_FILE = ${DOCKER_COMPOSE}
run-rasa-calm-demo-test-containers: BOT_PATH = $(RASA_CALM_DEMO_BOT_SETUP_PATH)/rasa-calm-demo-bot
run-rasa-calm-demo-test-containers: ## Run the containers.
	$(RUN_RASA_CALM_DEMO_CONTAINERS_COMMAND)

TEST_ENTERPRISE_SEARCH_INTEGRATION_COMMAND = poetry run \
		pytest $(ENTERPRISE_SEARCH_TEST_PATH) \
		-n $(JOBS) \
		--junitxml=$(RESULTS_FILE)



# Run the enterprise search integration tests with CALM bot
test-enterprise-search-integration-with-calm-bot: ENTERPRISE_SEARCH_TEST_PATH = $(ENTERPRISE_SEARCH_INTEGRATION_TEST_PATH)/${TEST_NAME}
test-enterprise-search-integration-with-calm-bot: RESULTS_FILE = integration-results-enterprise-search.xml
test-enterprise-search-integration-with-calm-bot:  ## Run the enterprise search integration tests with CALM bot.
	$(TEST_ENTERPRISE_SEARCH_INTEGRATION_COMMAND)

stop-rasa-calm-demo-bot-test-containers: DOCKER_COMPOSE_FILE = ${DOCKER_COMPOSE}
stop-rasa-calm-demo-bot-test-containers: ## Stop the metrics integration test containers.
	$(STOP_RASA_CALM_DEMO_CONTAINERS)

TEST_CHANNEL_CONNECTOR_INTEGRATION_COMMAND = poetry run \
        pytest $(CHANNEL_CONNECTOR_TEST_PATH) \
        -n $(JOBS) \
        --junitxml=$(RESULTS_FILE) \
        --ignore $(VOICE_READY_CONNECTOR_INTEGRATION_TEST_PATH) \
        --ignore $(VOICE_STREAM_CONNECTOR_INTEGRATION_TEST_PATH)

RUN_CHANNEL_CONNECTOR_CONTAINER_COMMAND = USER_ID=$(USER_ID) \
        docker compose \
        -f $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_channel_connectors/docker-compose.yml \
        up --wait
STOP_CHANNEL_CONNECTOR_CONTAINER_COMMAND = docker compose \
        -f $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_channel_connectors/docker-compose.yml \
        down

train-channel-connectors-calm-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=${RASA_PRO_LICENSE} -e OPENAI_API_KEY=${OPENAI_API_KEY}
train-channel-connectors-calm-bot: CONTAINER_NAME = rasa-channel-connectors-calm-bot-$(RASA_IMAGE_TAG)
train-channel-connectors-calm-bot: BOT_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_channel_connectors/calm-demo-bot
train-channel-connectors-calm-bot: ## Train the CALM bot for channel connectors integration tests.
	$(TRAIN_BOT_COMMAND)


run-channel-connectors-integration-containers: train-channel-connectors-calm-bot
	$(RUN_CHANNEL_CONNECTOR_CONTAINER_COMMAND)

# Run the channel connectors integration tests with CALM bot
test-channel-connectors-integration-with-calm-bot: CHANNEL_CONNECTOR_TEST_PATH = $(CHANNEL_CONNECTOR_INTEGRATION_TEST_PATH)
test-channel-connectors-integration-with-calm-bot: RESULTS_FILE = integration-results-channel-connectors-with-calm-bot-results.xml
test-channel-connectors-integration-with-calm-bot: ## Run the channel connectors integration tests with CALM bot.
	$(TEST_CHANNEL_CONNECTOR_INTEGRATION_COMMAND)

# Run the Custom Broker integration tests with CALM bot
test-custom-broker-integration-with-calm-bot: 
	poetry run \
        pytest tests/integration_tests/core/custom_components/test_custom_broker.py \
        -n $(JOBS) \
        --junitxml=integration-results-custom-broker-with-calm-bot-results.xml

stop-channel-connectors-integration-containers: ## Stop the channel connectors integration test containers.
	$(STOP_CHANNEL_CONNECTOR_CONTAINER_COMMAND)

MONGODB_DOCKER_COMPOSE_FILE_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_tracker_stores/mongo_db_tracker_store/docker-compose.mongodb.yml

RUN_MONGODB_CONTAINER_COMMAND = docker compose \
		-f $(MONGODB_DOCKER_COMPOSE_FILE_PATH) \
		up --wait

STOP_MONGODB_CONTAINER_COMMAND = docker compose \
		-f $(MONGODB_DOCKER_COMPOSE_FILE_PATH) \
		down

run-mongodb-container: ## Run the MongoDB container.
	$(RUN_MONGODB_CONTAINER_COMMAND)

stop-mongodb-container: ## Stop the MongoDB container.
	$(STOP_MONGODB_CONTAINER_COMMAND)

test-mongodb-tracker-store:  ## Run the MongoDB tracker store integration tests. Make sure to run run-mongodb-container before running this target.
	poetry run \
		pytest $(TRACKER_STORE_INTEGRATION_TEST_PATH)/test_mongo_tracker_store.py \
			-n $(JOBS) \
			--junitxml=integration-results-mongo-tracker-store.xml

run-otel-collector: ## Run OTEL collector, which would recieve traces and metrics, and export them to OTEL monitoring backend
	docker compose -f data/test_config/providers/otel-docker-compose.yml up --wait --remove-orphans

print-otel-collector-logs: ## Print OTEL collector logs on console
	docker logs otel-collector

stop-otel-collector: ## Stop OTEL collector
	docker compose -f data/test_config/providers/otel-docker-compose.yml down --remove-orphans

DYNAMO_DOCKER_COMPOSE_FILE_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_tracker_stores/dynamo_tracker_store/docker-compose.dynamo.yml

RUN_DYNAMO_CONTAINER_COMMAND = docker compose \
		-f $(DYNAMO_DOCKER_COMPOSE_FILE_PATH) \
		up --wait

STOP_DYNAMO_CONTAINER_COMMAND = docker compose \
		-f $(DYNAMO_DOCKER_COMPOSE_FILE_PATH) \
		down

run-dynamo-container: ## Run the Dynamo container.
	$(RUN_DYNAMO_CONTAINER_COMMAND)

stop-dynamo-container: ## Stop the Dynamo container.
	$(STOP_DYNAMO_CONTAINER_COMMAND)

test-dynamo-tracker-store:  ## Run the dynamo tracker store integration tests. Make sure to run run-dynamo-container before running this target.
	poetry run \
		pytest $(TRACKER_STORE_INTEGRATION_TEST_PATH)/test_dynamo_tracker_store.py \
			-n $(JOBS) \
			--junitxml=integration-results-dynamo-tracker-store.xml

TRAIN_PII_BOT_COMMAND = docker run --rm \
		-u $(USER_ID) \
		--name $(CONTAINER_NAME) \
		$(DOCKER_ENV_VARS) \
		-v $(BOT_PATH)\:/app/bot \
		-w /app/bot \
		$(RASA_REPOSITORY):$(RASA_IMAGE_TAG) \
		train --fixed-model-name $(MODEL_NAME)

PII_INTEGRATION_TESTS_DEPLOYMENT_PATH = $(INTEGRATION_TEST_DEPLOYMENT_PATH)/integration_tests_pii_management_in_calm
PII_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH = $(PII_INTEGRATION_TESTS_DEPLOYMENT_PATH)/docker-compose.yml
PII_INTEGRATION_TESTS_ENV_FILE = $(PII_INTEGRATION_TESTS_DEPLOYMENT_PATH)/env-file
PII_CALM_BOT_DIRECTORY = calm_demo_bot
train-pii-calm-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=$(RASA_PRO_LICENSE) -e OPENAI_API_KEY=$(OPENAI_API_KEY) -e HF_TOKEN=$(HF_TOKEN)
train-pii-calm-bot: CONTAINER_NAME = rasa-pro-training-calm-bot-$(RASA_IMAGE_TAG)
train-pii-calm-bot: BOT_PATH = $(PII_INTEGRATION_TESTS_DEPLOYMENT_PATH)/$(PII_CALM_BOT_DIRECTORY)
train-pii-calm-bot: ## Train the CALM bot for PII integration tests.
	$(TRAIN_PII_BOT_COMMAND)

RUN_PII_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(PII_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(PII_INTEGRATION_TESTS_ENV_FILE) \
		up --wait

run-pii-calm-containers: BOT_PATH = "./$(PII_CALM_BOT_DIRECTORY)" ## Run the PII integration test containers for CALM bot.
run-pii-calm-containers: train-pii-calm-bot
	$(RUN_PII_CONTAINERS_COMMAND)

STOP_PII_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(PII_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(PII_INTEGRATION_TESTS_ENV_FILE) \
		down

stop-pii-calm-containers: BOT_PATH = "./$(PII_CALM_BOT_DIRECTORY)" ## Stop the PII integration test containers.
stop-pii-calm-containers: ## Stop the PII integration test containers for CALM bot.
	$(STOP_PII_CONTAINERS_COMMAND)


TEST_PII_INTEGRATION_COMMAND = poetry run \
		pytest $(INTEGRATION_TEST_PATH) \
		-n $(JOBS) \
		--junitxml=$(RESULTS_FILE)

# Run the PII integration tests with CALM bot
test-pii-integration-with-calm-bot: INTEGRATION_TEST_PATH = $(CALM_PII_INTEGRATION_TEST_PATH)
test-pii-integration-with-calm-bot: RESULTS_FILE = pii-management-in-calm-integration-results.xml
test-pii-integration-with-calm-bot:  ## Run the pii integration tests with CALM bot.
	$(TEST_PII_INTEGRATION_COMMAND)
