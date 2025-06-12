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

### E2E Test on Bot with CALM supported providers
This workflow runs a E2E test(s) on a bot, using the various CALM supported LLM 
and Embeddings providers, to verify CALM's integration/compatibility with the 
providers' APIs. 

This workflow runs as a cron job at 1am UTC every Sunday, and can 
also be triggered manually/on-demand, and test results' notification is sent to 
`#dev-tribe-alerts` Slack channel, and traces and metrics from Rasa components are sent to [`Honeycomb`](https://ui.honeycomb.io/rasa/environments/engine). In case of failure, test results are saved, and can be
found on GitHub's web interface, in the `Artifacts` section of the `Action`'s run summary page.

### E2E Test on Rasa-Calm-Demo with supported providers and specified config
This workflow runs a single E2E test (with assertions) on the `rasa-calm-demo` 
bot (from its `passing/happy_path` category), using the specified CALM supported LLM 
or Embeddings provider (with option to also specify specific model to be used from the provider), 
to verify CALM's integration/compatibility with the providers' APIs.

This workflow can be triggered manually/on-demand, and test results' notification is sent to 
`#dev-tribe-alerts` Slack channel, and traces and metrics from Rasa components are sent to [`Honeycomb`](https://ui.honeycomb.io/rasa/environments/engine). In case of failure, test results are saved, and can be
found on GitHub's web interface, in the `Artifacts` section of the `Action`'s run summary page.

### Dialogue Understanding Tests
This workflow runs a Dialogue Understanding tests on the `rasa-calm-demo` 
bot, using the specified CALM supported LLM or Embeddings provider (with option to also specify specific model to be used from the provider), 
to evaluate Command Generator.

This workflow can be triggered manually/on-demand, and test results' notification is sent to 
`#dev-tribe-alerts` Slack channel, and traces and metrics from Rasa components are sent to [`Honeycomb`](https://ui.honeycomb.io/rasa/environments/engine). In case of failure, test results are saved, and can be
found on GitHub's web interface, in the `Artifacts` section of the `Action`'s run summary page.

### DM1/Tensorflow tests
This workflow runs DM1 tests (that use Tensorflow), on:
1. Pull-requests and push to `main` and non-dev `release` branches and tags, when files containing DM1/Tensorflow code are changed. 
    - The list of files and associated tests can be found [here](https://www.notion.so/rasa/ENG-1615-Identify-DM1-TensorFlow-files-and-tests-190b9c0d544a80198b11c70d55610273?pvs=4).
2. Preparation of non-dev release branches.

Purpose of this dedicated workflow is to reduce GHA costs, by not running DM1/Tensorflow tests on all CI runs.

### Verify minor and patch releases are backwards compatible
This workflow runs on release branches, to verify compatibility between minor and patch versions, by
training model on an earlier `rasa-pro` version and starting a user session, then continuing the active user session with newer `rasa-pro` version; to verify that:
1. Newer versions can run inference via old model without retraining.
2. Newer versions can load active sessions from previous version, and continue conversation in them. 

### Rasa versions performance testing
Automated performance evaluation of new Rasa-Pro versions' command generation against baseline of previous versions, for release testing and on changes to prompt templates, to check against performance degradations.

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

### Find Newly Added Dependencies

This workflow is run to check whether new direct dependencies were added to `pyproject.toml`.
If new dependencies were added, the workflow will comment on the PR with the newly added dependencies to inform the developer
that the dependabot configuration must also be updated for the newly added dependencies.

## Capture Installation Time On PR Branches

This workflow is run to check whether the `poetry.lock` has been updated to detect PRs where dependency updates are being
made. If the `poetry.lock` file has been updated, the workflow will dispatch an event to run the `Run Performance Checks` workflow.
The `Run Performance Checks` workflow will publish a comment on the PR with the installation time of the rasa-pro package with the updated dependencies.

### Run Performance Checks
This workflow is run to check the rasa-pro package installation performance. It gathers metrics such as, `installation time`,
`commit time`, `rasa pro version` and sends this data to segment. 
Workflow runs once at the end of each day against the `main` branch.
Failure information is sent via Slack notification to the channel `#atom-squad-alerts`.
This workflow can also be run as a [workflow dispatch](#glossary) event or as a [repository dispatch](#glossary) event.
When run as a repository dispatch event, it will comment on the event source PR with the installation time of the rasa-pro package.

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
- repository dispatch : A repository dispatch event is an event that triggers a GitHub Actions workflow in a repository. For
more information, refer [here](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#repository_dispatch)
