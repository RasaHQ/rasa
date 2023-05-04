# Tests
This is an entire set of all tests for the project. 
It is divided into several parts: 
    * unit tests and integration tests
    * regression tests
    * acceptance tests

## Unit tests and integration tests
These are executed by our CI for every Pull Request.
They are located in all directories except `tests/regression` and `tests/acceptance`.

## Regression tests
These are executed by our CI for every Pull Request.
They are located in the `tests/regression` directory.

## Acceptance tests
These are executed by our CI before every release.
They are located in the `tests/acceptance` directory.