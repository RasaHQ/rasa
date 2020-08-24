.PHONY: clean test lint init docs

JOBS ?= 1

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
	@echo "        Lint code with flake8, and check if black formatter should be applied."
	@echo "    types"
	@echo "        Check for type errors using pytype."
	@echo "    prepare-tests-ubuntu"
	@echo "        Install system requirements for running tests on Ubuntu and Debian based systems."
	@echo "    prepare-tests-macos"
	@echo "        Install system requirements for running tests on macOS."
	@echo "    prepare-tests-windows"
	@echo "        Install system requirements for running tests on Windows."
	@echo "    prepare-tests-files"
	@echo "        Download all additional project files needed to run tests."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    livedocs"
	@echo "        Build the docs locally."
	@echo "    release"
	@echo "        Prepare a release."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .pytype/
	rm -rf dist/
	rm -rf docs/build
	rm -rf docs/.docusaurus

install:
	poetry run python -m pip install -U pip
	poetry install

install-mitie:
	poetry run python -m pip install -U git+https://github.com/tmbo/MITIE.git#egg=mitie

install-full: install install-mitie
	poetry install -E full

install-full-windows: install install-mitie
	# tensorflow_text is not available on Windows, so we're skipping it
	# see https://github.com/tensorflow/text/issues/44 for more details
	poetry install -E spacy -E transformers -E jieba

install-docs:
	cd docs/ && yarn install

formatter:
	poetry run black rasa tests

lint:
	poetry run flake8 rasa tests
	poetry run black --check rasa tests

types:
	poetry run pytype --keep-going rasa -j 16

prepare-tests-files:
	poetry install -E spacy
	poetry run python -m spacy download en_core_web_md
	poetry run python -m spacy download de_core_news_sm
	poetry run python -m spacy link en_core_web_md en --force
	poetry run python -m spacy link de_core_news_sm de --force
	wget --progress=dot:giga -N -P data/ https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat

prepare-wget-macos:
	brew install wget || true

prepare-wget-windows:
	choco install wget

prepare-tests-macos: prepare-wget-macos prepare-tests-files
	brew install graphviz || true

prepare-tests-ubuntu: prepare-tests-files
	sudo apt-get -y install graphviz graphviz-dev python-tk

prepare-tests-windows: prepare-wget-windows prepare-tests-files
	choco install graphviz

test: clean
	# OMP_NUM_THREADS can improve overall performance using one thread by process (on tensorflow), avoiding overload
	OMP_NUM_THREADS=1 poetry run pytest tests -n $(JOBS) --cov rasa

generate-pending-changelog:
	poetry run python -c "from scripts import release; release.generate_changelog('major.minor.patch')"

cleanup-generated-changelog:
	# this is a helper to cleanup your git status locally after running "make test-docs"
	# it's not run on CI at the moment
	git status --porcelain | sed -n '/^D */s///p' | xargs git reset HEAD
	git reset HEAD CHANGELOG.mdx
	git ls-files --deleted | xargs git checkout
	git checkout CHANGELOG.mdx

test-docs: generate-pending-changelog docs
	poetry run pytest tests/docs/*

docs:
	cd docs/ && poetry run yarn pre-build && yarn build

livedocs:
	cd docs/ && poetry run yarn start

release:
	poetry run python scripts/release.py
