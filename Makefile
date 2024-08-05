.PHONY: clean test lint init docs format formatter build-docker

JOBS ?= 1
INTEGRATION_TEST_FOLDER = tests/integration_tests
INTEGRATION_TEST_PYTEST_MARKERS ?= "sequential or broker or concurrent_lock_store or ((not sequential) and (not broker) and (not concurrent_lock_store))"
PLATFORM ?= "linux/arm64"
TRACING_INTEGRATION_TEST_FOLDER = $(INTEGRATION_TEST_FOLDER)/tracing
METRICS_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/tracing/test_metrics.py
CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(INTEGRATION_TEST_FOLDER)/core/actions/custom_actions
NLU_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)/test_custom_actions_with_nlu.py
CALM_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH = $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)/test_custom_actions_with_calm.py
BASE_IMAGE_HASH ?= localdev
BASE_BUILDER_IMAGE_HASH ?= localdev
RASA_DEPS_IMAGE_HASH ?= localdev
IMAGE_TAG ?= rasa-private:rasa-private-dev
POETRY_VERSION ?= 1.8.2
BOT_PATH ?=
MODEL_NAME ?= model
# find user's id
USER_ID := $(shell id -u)

help:
	@echo "make"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
	@echo "    install"
	@echo "        Install rasa."
	@echo "    install-full"
	@echo "        Install rasa with all extras (transformers, tensorflow_text, spacy, jieba)."
	@echo "    formatter"
	@echo "        Apply ruff formatting to code."
	@echo "    lint"
	@echo "        Lint code with ruff, and check if ruff formatter should be applied."
	@echo "    lint-docstrings"
	@echo "        Check docstring conventions in changed files."
	@echo "    types"
	@echo "        Check for type errors using mypy."
	@echo "    static-checks"
	@echo "        Run all python static checks."
	@echo "    prepare-tests-ubuntu"
	@echo "        Install system requirements for running tests on Ubuntu and Debian based systems."
	@echo "    prepare-tests-macos"
	@echo "        Install system requirements for running tests on macOS."
	@echo "    prepare-tests-windows"
	@echo "        Install system requirements for running tests on Windows."
	@echo "    prepare-spacy"
	@echo "        Download models needed for spacy tests."
	@echo "    prepare-mitie"
	@echo "        Download the standard english mitie model."
	@echo "    prepare-transformers"
	@echo "        Download all models needed for testing LanguageModelFeaturizer."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    test-integration"
	@echo "        Run integration tests using pytest."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    livedocs"
	@echo "        Build the docs locally."
	@echo "    release"
	@echo "        Prepare a release."
	@echo "    build-docker"
	@echo "        Build Rasa Pro Docker image."
	@echo "    run-integration-containers"
	@echo "        Run the integration test containers."
	@echo "    stop-integration-containers"
	@echo "        Stop the integration test containers."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf dist/

install:
	poetry run python -m pip install -U pip
	poetry install

install-mitie:
	poetry run python -m pip install -U pip
	poetry run python -m pip install -U git+https://github.com/tmbo/MITIE.git#egg=mitie

install-full: install-mitie
	poetry install -E full

formatter:
	poetry run ruff format rasa tests

format: formatter

lint:
     # Ignore docstring errors when running on the entire project
	poetry run ruff check rasa tests --ignore D
	poetry run ruff format --check rasa tests
	make lint-docstrings

# Compare against `main` if no branch was provided
BRANCH ?= main
lint-docstrings:
	./scripts/lint_python_docstrings.sh $(BRANCH)

lint-changelog:
	./scripts/lint_changelog_files.sh

lint-security:
	poetry run bandit -ll -ii -r --config pyproject.toml rasa/*

types:
	poetry run mypy rasa

static-checks: lint lint-security types

prepare-spacy:
	poetry run python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
	poetry run python -m pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl

prepare-mitie:
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

prepare-transformers:
	while read -r MODEL; do poetry run python scripts/download_transformer_model.py $$MODEL ; done < data/test/hf_transformers_models.txt
	if ! [ $(CI) ]; then poetry run python scripts/download_transformer_model.py rasa/LaBSE; fi
prepare-tests-macos:
	brew install wget graphviz || true

prepare-tests-ubuntu:
	sudo apt-get -y install graphviz graphviz-dev python-tk

prepare-tests-windows:
	choco install wget graphviz

# GitHub Action has pre-installed a helper function for installing Chocolatey packages
# It will retry the installation 5 times if it fails
# See: https://github.com/actions/virtual-environments/blob/main/images/win/scripts/ImageHelpers/ChocoHelpers.ps1
prepare-tests-windows-gha:
	powershell -command "Install-ChocoPackage wget graphviz"

test: clean
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

test-integration:
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
ifeq (,$(wildcard tests_deployment/.env))
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest $(INTEGRATION_TEST_FOLDER)/ \
			-n $(JOBS) \
			-m $(INTEGRATION_TEST_PYTEST_MARKERS) \
			--dist loadgroup  \
			--ignore $(TRACING_INTEGRATION_TEST_FOLDER) \
			--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
			--junitxml=report_integration.xml
else
	set -o allexport; \
	source tests_deployment/.env && \
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest $(INTEGRATION_TEST_FOLDER)/ \
			-n $(JOBS) \
			-m $(INTEGRATION_TEST_PYTEST_MARKERS) \
			--dist loadgroup \
			--ignore $(TRACING_INTEGRATION_TEST_FOLDER) \
			--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
			--junitxml=report_integration.xml && \
	set +o allexport
endif

test-anonymization: PYTEST_MARKER=category_anonymization and (not flaky) and (not acceptance)
test-anonymization: DD_ARGS := $(or $(DD_ARGS),)
test-anonymization: test-marker

test-cli: PYTEST_MARKER=category_cli and (not flaky) and (not acceptance) and (not category_anonymization)
test-cli: DD_ARGS := $(or $(DD_ARGS),)
test-cli: test-marker

test-core-featurizers: PYTEST_MARKER=category_core_featurizers and (not flaky) and (not acceptance) and (not category_anonymization)
test-core-featurizers: DD_ARGS := $(or $(DD_ARGS),)
test-core-featurizers: test-marker

test-policies: PYTEST_MARKER=category_policies and (not flaky) and (not acceptance) and (not category_anonymization)
test-policies: DD_ARGS := $(or $(DD_ARGS),)
test-policies: test-marker

test-nlu-featurizers: PYTEST_MARKER=category_nlu_featurizers and (not flaky) and (not acceptance) and (not category_anonymization)
test-nlu-featurizers: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-featurizers: prepare-spacy prepare-mitie prepare-transformers test-marker

test-nlu-predictors: PYTEST_MARKER=category_nlu_predictors and (not flaky) and (not acceptance) and (not category_anonymization)
test-nlu-predictors: DD_ARGS := $(or $(DD_ARGS),)
test-nlu-predictors: prepare-spacy prepare-mitie test-marker

test-full-model-training: PYTEST_MARKER=category_full_model_training and (not flaky) and (not acceptance) and (not category_anonymization)
test-full-model-training: DD_ARGS := $(or $(DD_ARGS),)
test-full-model-training: prepare-spacy prepare-mitie prepare-transformers test-marker

test-other-unit-tests: PYTEST_MARKER=category_other_unit_tests and (not flaky) and (not acceptance) and (not category_anonymization)
test-other-unit-tests: DD_ARGS := $(or $(DD_ARGS),)
test-other-unit-tests: prepare-spacy prepare-mitie test-marker

test-performance: PYTEST_MARKER=category_performance and (not flaky) and (not acceptance) and (not category_anonymization)
test-performance: DD_ARGS := $(or $(DD_ARGS),)
test-performance: test-marker

test-flaky: PYTEST_MARKER=flaky and (not acceptance) and (not category_anonymization)
test-flaky: DD_ARGS := $(or $(DD_ARGS),)
test-flaky: prepare-spacy prepare-mitie test-marker

test-acceptance: PYTEST_MARKER=acceptance and (not flaky) and (not category_anonymization)
test-acceptance: DD_ARGS := $(or $(DD_ARGS),)
test-acceptance: prepare-spacy prepare-mitie test-marker

test-gh-actions:
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest .github/tests --cov .github/scripts

test-marker: clean
    # OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
	TRANSFORMERS_OFFLINE=1 \
	OMP_NUM_THREADS=1 \
	TF_CPP_MIN_LOG_LEVEL=2 \
	poetry run \
		pytest tests \
			-n $(JOBS) \
			--dist loadscope \
			-m "$(PYTEST_MARKER)" \
			--cov rasa \
			--ignore $(INTEGRATION_TEST_FOLDER)/ $(DD_ARGS)

release:
	poetry run python scripts/release.py prepare --interactive

build-docker-base:
	docker build . -t rasa-private:base-localdev -f docker/Dockerfile.base --progress=plain --platform=$(PLATFORM)

build-docker-builder:
	docker build . -t rasa-private:base-builder-localdev -f docker/Dockerfile.base-builder --build-arg IMAGE_BASE_NAME=rasa-private --build-arg BASE_IMAGE_HASH=$(BASE_IMAGE_HASH) --progress=plain --platform=$(PLATFORM)

build-docker-rasa-deps:
	docker build . -t rasa-private:rasa-deps-localdev -f docker/Dockerfile.rasa-deps --build-arg IMAGE_BASE_NAME=rasa-private --build-arg BASE_BUILDER_IMAGE_HASH=$(BASE_BUILDER_IMAGE_HASH) --build-arg POETRY_VERSION=$(POETRY_VERSION) --progress=plain --platform=$(PLATFORM)

build-docker-rasa-image:
	docker build . -t $(IMAGE_TAG) -f Dockerfile --build-arg IMAGE_BASE_NAME=rasa-private --build-arg BASE_IMAGE_HASH=$(BASE_IMAGE_HASH) --build-arg RASA_DEPS_IMAGE_HASH=$(RASA_DEPS_IMAGE_HASH) --progress=plain --platform=$(PLATFORM)

build-docker: build-docker-base build-docker-builder build-docker-rasa-deps build-docker-rasa-image

build-tests-deployment-env: ## Create environment files (.env) for docker-compose.
	cd tests_deployment && \
	test -f .env || cat .env.example >> .env

run-integration-containers: build-tests-deployment-env ## Run the integration test containers.
	cd tests_deployment && \
	docker compose -f docker-compose.integration.yml up &

stop-integration-containers: ## Stop the integration test containers.
	cd tests_deployment && \
	docker compose -f docker-compose.integration.yml down

tag-release-auto:
	poetry run python scripts/release.py tag --skip-confirmation

tests_deployment/integration_tests_tracing_deployment/simple_bot/models/model.tar.gz:
	cd ./tests_deployment/integration_tests_tracing_deployment/simple_bot && poetry run rasa train --fixed-model-name model

train: tests_deployment/integration_tests_tracing_deployment/simple_bot/models/model.tar.gz

run-tracing-integration-containers: train ## Run the tracing integration test containers.
	docker-compose -f tests_deployment/integration_tests_tracing_deployment/docker-compose.intg.yml up -d

stop-tracing-integration-containers: ## Stop the tracing integration test containers.
	docker-compose -f tests_deployment/integration_tests_tracing_deployment/docker-compose.intg.yml down

test-tracing-integration:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	PYTHONPATH=./vendor/jaeger-python-proto \
	poetry run \
		pytest $(TRACING_INTEGRATION_TEST_FOLDER) \
		-n $(JOBS) \
		--ignore $(METRICS_INTEGRATION_TEST_PATH) \
		--ignore $(CUSTOM_ACTIONS_INTEGRATION_TEST_PATH) \
		--junitxml=integration-results-tracing.xml

train-calm:
	cd ./tests_deployment/integration_tests_tracing_deployment/metrics_setup/calm_bot && poetry run rasa train --fixed-model-name model

run-metrics-integration-containers: train-calm ## Run the metrics integration test containers.
	docker compose -f tests_deployment/integration_tests_tracing_deployment/metrics_setup/docker-compose.yml up -d --wait

stop-metrics-integration-containers:
	docker compose -f tests_deployment/integration_tests_tracing_deployment/metrics_setup/docker-compose.yml down

test-metrics-integration:
	poetry run pytest $(METRICS_INTEGRATION_TEST_PATH) -n $(JOBS) --junitxml=integration-results-metric.xml


RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_PATH = $(PWD)/tests_deployment/integration_tests_custom_action_server
RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH = $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_PATH)/docker-compose.yml
RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE = $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_PATH)/env-file
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
		${RASA_REPOSITORY}:${RASA_IMAGE_TAG} \
		train --fixed-model-name $(MODEL_NAME)

train-action-server-nlu-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=${RASA_PRO_LICENSE}
train-action-server-nlu-bot: CONTAINER_NAME = rasa-pro-training-nlu-bot-${RASA_IMAGE_TAG}
train-action-server-nlu-bot: BOT_PATH = $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_PATH)/$(NLU_BOT_DIRECTORY)
train-action-server-nlu-bot:
	$(TRAIN_BOT_COMMAND)


train-action-server-calm-bot: DOCKER_ENV_VARS = -e RASA_PRO_LICENSE=${RASA_PRO_LICENSE} -e OPENAI_API_KEY=${OPENAI_API_KEY}
train-action-server-calm-bot: CONTAINER_NAME = rasa-pro-training-calm-bot-${RASA_IMAGE_TAG}
train-action-server-calm-bot: BOT_PATH = $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_PATH)/$(CALM_BOT_DIRECTORY)
train-action-server-calm-bot:
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
		-f $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE) \
		up --wait

# This target is mutually exclusive with run-action-server-calm-containers
run-action-server-nlu-containers: BOT_PATH = "./${NLU_BOT_DIRECTORY}"
run-action-server-nlu-containers: train-action-server-nlu-bot
	$(RUN_ACTION_SERVER_CONTAINERS_COMMAND)


# This target is mutually exclusive with run-action-server-nlu-containers
run-action-server-calm-containers: BOT_PATH = "./${CALM_BOT_DIRECTORY}"
run-action-server-calm-containers: train-action-server-calm-bot
	$(RUN_ACTION_SERVER_CONTAINERS_COMMAND)


# Command to stop the action server containers.
# This command is used by both NLU and CALM to stop action server targets.
# Reason for not using a target is that we need to pass the BOT_PATH variable in two different targets
# and if in the future we want to stop both NLU and CALM action server containers in one command,
# we cannot reuse the target with different variables.
STOP_ACTION_SERVER_CONTAINERS_COMMAND = USER_ID=$(USER_ID) \
	BOT_PATH=$(BOT_PATH) \
	docker compose \
		-f $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_DOCKER_COMPOSE_PATH) \
		--env-file $(RASA_CUSTOM_ACTION_SERVER_INTEGRATION_TESTS_ENV_FILE) \
		down

# Stop the action server integration test containers.
stop-action-server-nlu-containers: BOT_PATH = "./${NLU_BOT_DIRECTORY}"
stop-action-server-nlu-containers:
	$(STOP_ACTION_SERVER_CONTAINERS_COMMAND)

stop-action-server-calm-containers: BOT_PATH = "./${CALM_BOT_DIRECTORY}"
stop-action-server-calm-containers:
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
test-custom-action-integration-with-nlu-bot:
	$(TEST_CUSTOM_ACTION_INTEGRATION_COMMAND)


# Run the action server integration tests with CALM bot
test-custom-action-integration-with-calm-bot: INTEGRATION_TEST_PATH = $(CALM_CUSTOM_ACTIONS_INTEGRATION_TEST_PATH)
test-custom-action-integration-with-calm-bot: RESULTS_FILE = integration-results-custom-actions-with-calm-bot-results.xml
test-custom-action-integration-with-calm-bot:
	$(TEST_CUSTOM_ACTION_INTEGRATION_COMMAND)
