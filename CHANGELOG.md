# Change Log
All notable changes to this project will be documented in this file. 
This project adheres to [Semantic Versioning](http://semver.org/) starting with version 0.7.0.

## [Unreleased]
### Added
### Changed
### Removed
### Fixed

## [0.7.0] - 2017-02-28
### Added
- option to use multi-threading during classifier training
- entity synonym support 
- proper temporary file creation during tests
- mitie_sklearn backend using mitie tokenization and sklearn classification
- option to fine-tune spacy NER models
- multithreading support of build in REST server (e.g. using gunicorn)
### Fixed
- error propagation on failed vector model loading (spacy)
- escaping of special characters during mitie tokenization

## [0.6-beta] - 2017-01-31