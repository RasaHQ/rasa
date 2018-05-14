.PHONY: clean test lint

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

livedocs:
	cd docs && make livehtml

check-readme:
	# if this runs through we can be sure the readme is properly shown on pypi
	python setup.py check --restructuredtext --strict
