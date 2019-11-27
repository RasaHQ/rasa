.PHONY: clean test lint init check-readme

JOBS ?= 1

help:
	@echo "make"
	@echo "    clean"
	@echo "        Remove Python/build artifacts."
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
	@echo "    prepare-tests-files"
	@echo "        Download all additional project files needed to run tests."
	@echo "    test"
	@echo "        Run pytest on tests/."
	@echo "        Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo "    check-readme"
	@echo "        Check if the README can be converted from .md to .rst for PyPI."
	@echo "    doctest"
	@echo "        Run all doctests embedded in the documentation."
	@echo "    livedocs"
	@echo "        Build the docs locally."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .pytype/
	rm -rf dist/
	rm -rf docs/_build

formatter:
	black rasa tests

lint:
	flake8 rasa tests
	black --check rasa tests

types:
	pytype --keep-going rasa

prepare-tests-macos: prepare-tests-files
	brew install graphviz wget || true

prepare-tests-ubuntu: prepare-tests-files
	sudo apt-get -y install graphviz graphviz-dev python3-tk

prepare-tests-files:
	pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.1.0/en_core_web_md-2.1.0.tar.gz#egg=en_core_web_md==2.1.0 --no-cache-dir -q
	python -m spacy link en_core_web_md en --force
	pip3 install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.1.0/de_core_news_sm-2.1.0.tar.gz#egg=de_core_news_sm==2.1.0 --no-cache-dir -q
	python -m spacy link de_core_news_sm de --force
	wget --progress=dot:giga -N -P data/ https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat

test: clean
	# OMP_NUM_THREADS can improve overral performance using one thread by process (on tensorflow), avoiding overload
	OMP_NUM_THREADS=1 pytest tests -n $(JOBS) --cov rasa

doctest: clean
	cd docs && make doctest

livedocs:
	cd docs && make livehtml

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext --strict
