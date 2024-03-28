.PHONY: clean test lint init docs format formatter build-docker

JOBS ?= 1
INTEGRATION_TEST_FOLDER = tests/integration_tests/
INTEGRATION_TEST_PYTEST_MARKERS ?= "sequential or broker or concurrent_lock_store or ((not sequential) and (not broker) and (not concurrent_lock_store))"
PLATFORM ?= "linux/amd64"
TRACING_INTEGRATION_TEST_FOLDER = tests/integration_tests/tracing
METRICS_INTEGRATION_TEST_PATH = tests/integration_tests/tracing/test_metrics.py

help:
	@echo "make"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
	@echo "    install"
	@echo "        Install rasa."
	@echo "    install-full"
	@echo "        Install rasa with all extras (transformers, tensorflow_text, spacy, jieba)."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Lint code with ruff, and check if black formatter should be applied."
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
	@echo "        Build Rasa Open Source Docker image."
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
	poetry run black rasa tests

format: formatter

lint:
     # Ignore docstring errors when running on the entire project
	poetry run ruff check rasa tests --ignore D
	poetry run black --check rasa tests
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
	poetry run python -m spacy download en_core_web_md
	poetry run python -m spacy download de_core_news_sm

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
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest tests -n $(JOBS) --dist loadscope --cov rasa --ignore $(INTEGRATION_TEST_FOLDER)

test-integration:
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	# TF_CPP_MIN_LOG_LEVEL=2 sets C code log level for tensorflow to error suppressing lower log events
ifeq (,$(wildcard tests_deployment/.env))
	OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest $(INTEGRATION_TEST_FOLDER) -n $(JOBS) -m $(INTEGRATION_TEST_PYTEST_MARKERS) --dist loadgroup  --ignore $(TRACING_INTEGRATION_TEST_FOLDER) --junitxml=report_sequential.xml
else
	set -o allexport; source tests_deployment/.env && OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest $(INTEGRATION_TEST_FOLDER) -n $(JOBS) -m $(INTEGRATION_TEST_PYTEST_MARKERS) --dist loadgroup --ignore $(TRACING_INTEGRATION_TEST_FOLDER) --junitxml=report_sequential.xml && set +o allexport
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
	TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 poetry run pytest tests -n $(JOBS) --dist loadscope -m "$(PYTEST_MARKER)" --cov rasa --ignore $(INTEGRATION_TEST_FOLDER) $(DD_ARGS)

release:
	poetry run python scripts/release.py prepare --interactive

build-docker:
    	# Build base image
	docker build . -t rasa-private:base-localdev -f docker/Dockerfile.base --platform=linux/amd64
    	# Build base poetry image
	docker build . -t rasa-private:base-poetry-localdev -f docker/Dockerfile.base-poetry --build-arg IMAGE_BASE_NAME=rasa-private --build-arg BASE_IMAGE_HASH=localdev --build-arg POETRY_VERSION=1.4.2 --platform=linux/amd64
    	# Build base builder image
	docker build . -t rasa-private:base-builder-localdev -f docker/Dockerfile.base-builder --build-arg IMAGE_BASE_NAME=rasa-private --build-arg POETRY_VERSION=localdev --platform=linux/amd64
    	# Build Rasa Private image
	docker build . -t rasa-private:rasa-private-dev -f Dockerfile --build-arg IMAGE_BASE_NAME=rasa-private --build-arg BASE_IMAGE_HASH=localdev --build-arg BASE_BUILDER_IMAGE_HASH=localdev --platform=linux/amd64

build-tests-deployment-env: ## Create environment files (.env) for docker-compose.
	cd tests_deployment && \
	test -f .env || cat .env.example >> .env

run-integration-containers: build-tests-deployment-env ## Run the integration test containers.
	cd tests_deployment && \
	docker-compose -f docker-compose.integration.yml up &

stop-integration-containers: ## Stop the integration test containers.
	cd tests_deployment && \
	docker-compose -f docker-compose.integration.yml down

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
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=./vendor/jaeger-python-proto poetry run pytest $(TRACING_INTEGRATION_TEST_FOLDER) -n $(JOBS) --ignore $(METRICS_INTEGRATION_TEST_PATH) --junitxml=report_tracing.xml

train-calm:
	cd ./tests_deployment/integration_tests_tracing_deployment/metrics_setup/calm_bot && poetry run rasa train --fixed-model-name model

run-metrics-integration-containers: train-calm ## Run the metrics integration test containers.
	docker compose -f tests_deployment/integration_tests_tracing_deployment/metrics_setup/docker-compose.yml up -d --wait

stop-metrics-integration-containers:
	docker compose -f tests_deployment/integration_tests_tracing_deployment/metrics_setup/docker-compose.yml down

test-metrics-integration:
	poetry run pytest $(METRICS_INTEGRATION_TEST_PATH) -n $(JOBS)
