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
	@echo "    lint-docstrings"
	@echo "        Check docstring conventions in changed files."
	@echo "    types"
	@echo "        Check for type errors using mypy."
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
	rm -rf .mypy_cache/
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

install-docs:
	cd docs/ && yarn install

formatter:
	poetry run black rasa tests

lint:
     # Ignore docstring errors when running on the entire project
	poetry run flake8 rasa tests --extend-ignore D
	poetry run black --check rasa tests
	make lint-docstrings

# Compare against `master` if no branch was provided
BRANCH ?= master
lint-docstrings:
# Lint docstrings only against the the diff to avoid too many errors.
# Check only production code. Ignore other flake errors which are captured by `lint`
# Diff of committed changes (shows only changes introduced by your branch
ifneq ($(strip $(BRANCH)),)
	git diff $(BRANCH)...HEAD -- rasa | poetry run flake8 --select D --diff
endif

	# Diff of uncommitted changes for running locally
	git diff HEAD -- rasa | poetry run flake8 --select D --diff

types:
	# FIXME: working our way towards removing these
	# see https://github.com/RasaHQ/rasa/pull/6470
	# the list below is sorted by the number of errors for each error code, in decreasing order
	poetry run mypy rasa --disable-error-code arg-type \
	--disable-error-code assignment \
	--disable-error-code var-annotated \
	--disable-error-code return-value \
	--disable-error-code union-attr \
	--disable-error-code override \
	--disable-error-code operator \
	--disable-error-code attr-defined \
	--disable-error-code index \
	--disable-error-code misc \
	--disable-error-code return \
	--disable-error-code call-arg \
	--disable-error-code type-var \
	--disable-error-code list-item \
	--disable-error-code has-type \
	--disable-error-code valid-type \
	--disable-error-code dict-item \
	--disable-error-code no-redef \
	--disable-error-code func-returns-value

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
	cd docs && yarn mdx-lint

docs:
	cd docs/ && poetry run yarn pre-build && yarn build

livedocs:
	cd docs/ && poetry run yarn start

release:
	poetry run python scripts/release.py
