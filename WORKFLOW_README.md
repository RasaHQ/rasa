## README Contents:
- [Active Workflows](#active-workflows)
- [Actions](#actions)

## Active Workflows

### Continuous Integration
This is the main CI file, running on every pull request and pushes to `main` branch and all `release branches`. 
It runs the following checks:
- Build docker dev images for each pull request and push to aws
- Check poetry lock is up to date
- Check inspector build
- Check code quality and linting
- Run all unit tests on OS Ubuntu
- Run all Integration Tests
- Publish Test reports on Github Actions

### Run Windows Tests On Main
This workflow runs all unit tests on OS Windows. It is scheduled to run once every day against the `main` branch.
If this workflow fails a Slack notification is sent to the channel `#atom-squad-alerts`.
This workflow can also be run as a [workflow dispatch](#glossary) event.

### Run Tests On Main
This runs PII language and model provider specific tests on every merge to `main` and `release branches`. This workflow can also be 
run as a [workflow dispatch](#glossary) event. 
If this workflow fails a Slack notification is sent to the channel `#atom-squad-alerts`.

### E2E Test on Rasa-Calm-Demo with supported providers
This workflow runs a single E2E test (with assertions) on the `rasa-calm-demo` 
bot (from its `passing/happy_path` category), using the various CALM supported LLM 
and Embeddings providers, to verify CALM's integration/compatibility with the 
providers' APIs. 

This workflow runs as a cron job at 1am UTC every Sunday, and can 
also be triggered manually/on-demand, and test results' notification is sent to 
`#dev-tribe-alerts` Slack channel. In case of failure, test results are saved, and can be
found on GitHub's web interface, in the `Artifacts` section of the `Action`'s run summary page.

### Release Artifacts Workflow
Workflow runs on a tag push to the `main` branch and `release branches`. 
This workflow does the following:
- Optionally, runs the ["E2E Test on Rasa-Calm-Demo with supported providers"](./.github/workflows/providers-e2e-tests.yml) workflow, if this workflow is triggered manually with the `run_e2e_test_on_providers` option selected.
- builds and pushes a docker image with the tag version to [GCP](https://console.cloud.google.com/artifacts/docker/rasa-releases/europe-west3/rasa-pro/rasa-pro?authuser=1&project=rasa-releases)
- builds and pushes a python package to [GCP](https://console.cloud.google.com/artifacts/python/rasa-releases/europe-west3/rasa-pro-python/rasa-pro?authuser=1&project=rasa-releases)

This workflow can also be run as a [workflow dispatch](#glossary) event.
If this workflow fails a Slack notification is sent to the channel `#devtribe`.

### Tag Release
Workflow pushes a tag by running `make tag-release-auto`.
This workflow runs on `main` and `release branches` and is triggered only after the release prep branch with the name `prepare-release*` is merged.

### Release
Workflow runs only on tag pushes and branches starting with `prepare-release*`. 
It checks if the tag version is a pre-release version or not and 
checks changelog folder to list any unexpected files.

### Backport
In order to backport changes to main and across release branches, we use the [backport-github-action](https://github.com/sorenlouv/backport-github-action) GitHub Action.
This GitHub Action backports the changes to the specified release branch(es) and assigns the original PR author as the reviewer.
The label `backport-to-main` should be applied to release PRs too to backport the `CHANGELOG.md` updates to `main`.
While the action will backport all release changes, including version updates, version updates should be accepted to
the `main` branch from the latest release branch only.

Note that the label should be applied before the source PR gets merged.
When a pull request gets labelled `backport-to-<release-branch>`, a pull request is opened by the `backport-github-action`
as soon as the source PR gets closed (by merging).

The configuration for this GitHub action can be found in the `.backportrc.json` file located in the root folder.
We have to update `targetBranchChoices` with every new release branch created after every minor or major.
The configuration allows PRs to be opened which might contain conflicts with the target branch: the PR author has to 
resolve any conflicts before approving and merging.

### Run Performance Checks On Main
This workflow is run to check the rasa-pro package installation performance. It gathers metrics such as, `installation time`,
`commit time`, `rasa pro version` and sends this data to segment. 
Workflow runs once at the end of each day against the `main` branch.
Failure information is sent via Slack notification to the channel `#atom-squad-alerts`.
This workflow can also be run as a [workflow dispatch](#glossary) event.

### Security Patching
Runs a security scan for vulnerabilities, uploads report to GCS and alerts slack bot on the findings.
Runs as a cron job at 8AM Monday to Sunday.
This workflow can also be run as a [workflow dispatch](#glossary) event.

### Security Scan Docker Image Dependencies
Scans all released rasa-pro docker images for dependency vulnerabilities, uploads scan reports to GCS and alerts slackbot on findings.
Runs as a cron job at 8AM Monday to Sunday.
This workflow can also be run as a [workflow dispatch](#glossary) event.

### Security Scans
Runs on pull request types `opened`, `synchronize`and `labeled`. Detects hardcoded secrets, runs a dependency vulnerabilities scanner and also
detects security issues with python dependencies.

### Semgrep Check
Scans for security issues using the Semgrep tool. Runs on `main` and pull requests.

### Git Backup
Back up rasa-private repo to S3. Runs as a cron job at 7AM Monday to Sunday. Can also be run as a [workflow dispatch](#glossary) event.

### PR Cleanup
This workflow runs everytime a pull request is closed and skips pull requests created by dependabot. 
Deletes all dev docker images created as part of the [continuous integration workflow](#continuous-integration)
and removes them from the ECR repositories.

## Actions
In order to remove duplications in the CI workflow steps actions were packaged using [composite actions](https://docs.github.com/en/actions/creating-actions/creating-a-composite-action).
The current set of actions are as follows :
- .actions/debug-broker-tests/action.yml - displays zookeeper and kafka logs when integration tests are run.
- .actions/debug-metric-tests/action.yml - displays action-server, otlp-collector, prometheus and rasa-pro-assistent container logs when integration tests are run.
- .actions/debug-tracing-tests/action.yml - displays otel-collector, jaegar, rasa-pro and action server container logs when integration tests are run.
- .actions/pull-from-ecr/action.yml - configures AWS and logs in to ECR and pulls rasa-pro dev image.
- .actions/setup-build-x/action.yml - sets up QEMU and docker buildx
- .actions/test-prerequisites/action.yml - Setups python, reads poetry version, installs poetry, load cached venv and installs setuptool when running unit/integration tests.

## Glossary
- workflow dispatch : Only workflow files that use the workflow_dispatch event trigger will have the option to run the workflow manually using the Run workflow button. 
 For more information refer [here](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_dispatch)
