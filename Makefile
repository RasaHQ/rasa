.PHONY: clean test lint init check-readme

TEST_PATH=./

help:
	@echo "    clean"
	@echo "        Remove python artifacts and build artifacts."
	@echo "    lint"
	@echo "        Check style with flake8."
	@echo "    test"
	@echo "        Run py.test"
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
	rm -rf docs/_build

lint:
	py.test --pep8 -m pep8

test: clean
	py.test tests --verbose --pep8 --color=yes $(TEST_PATH)

doctest: clean
	cd docs && make doctest

livedocs:
	cd docs && make livehtml

check-readme:
	# if this runs through we can be sure the readme is properly shown on pypi
	python setup.py check --restructuredtext --strict
