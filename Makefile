.PHONY: clean test lint init check-readme

help:
	@echo "    clean"
	@echo "        Remove python artifacts and build artifacts."
	@echo "    formatter"
	@echo "        Apply black formatting to code."
	@echo "    lint"
	@echo "        Check style with flake8."
	@echo "    types"
	@echo "        Check for type errors using pytype."
	@echo "    test"
	@echo "        Run py.test"
	@echo "    check-readme"
	@echo "        Check if the readme can be converted from md to rst for pypi"

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

# TODO: Remove '--exit-zero'
lint:
	flake8 rasa tests --exit-zero
	black --check rasa tests

types:
	pytype --keep-going rasa

test: clean
	py.test tests --cov rasa

doctest: clean
	cd docs && make doctest

livedocs:
	cd docs && make livehtml

# if this runs through we can be sure the readme is properly shown on pypi
check-readme:
	python setup.py check --restructuredtext --strict
