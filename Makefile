.PHONY: clean test lint init check-readme

TEST_PATH=./

help:
	@echo "    clean"
	@echo "        Remove python artifacts and build artifacts."
	@echo "    lint"
	@echo "        Check style with flake8."
	@echo "    test"
	@echo "        Run py.test"
	@echo "    check-readme"
	@echo "        Check if the readme can be converted from md to rst for pypi"
	@echo "    init"
	@echo "        Install Rasa Core"

init:
	pip install -r requirements.txt

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf docs/core/_build
	rm -rf docs/nlu/_build

lint:
	black .

test: clean
	py.test tests --verbose --color=yes $(TEST_PATH)
	black --check .

doctest: clean
	cd docs/core && make doctest

livedocs-core:
	cd docs/core && make livehtml

livedocs-nlu:
	cd docs/nlu && make livehtml

check-readme:
	# if this runs through we can be sure the readme is properly shown on pypi
	python setup.py check --restructuredtext --strict
