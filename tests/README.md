# Tests
This directory contains all tests for the projects.
Tests are organized into several groups:
* unit tests and integration tests
* regression tests
* acceptance tests

### Unit tests and integration tests
These are executed by our CI for every Pull Request.
They are located in all directories except `tests/regression` and `tests/acceptance_tests`.

### Regression tests
These are executed by our CI before every release.
They are located in the `tests/regressions` directory.

### Acceptance tests
These are executed by our CI before every release.
They are located in the `tests/acceptance_tests` directory.