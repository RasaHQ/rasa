<h1 align="center">Rasa Pro</h1>

<div align="center">

[![Build Status](https://github.com/RasaHQ/rasa-private/workflows/Continuous%20Integration/badge.svg)](https://github.com/RasaHQ/rasa-private/actions)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=RasaHQ_rasa&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=RasaHQ_rasa)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs/rasa-pro/)

</div>

<hr />

<img align="right" height="255" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png" alt="An image of Sara, the Rasa mascot bird, holding a flag that reads Open Source with one wing, and a wrench in the other" title="Rasa Pro">

Rasa Pro is an open core product that extends Rasa Open Source. With over 50 million downloads, Rasa Open Source is the most popular open source framework for building chat and voice-based AI assistants.

Rasa Pro introduces CALM, a generative AI-native approach to developing assistants, combined with enterprise-ready analytics, security, and observability capabilities. A paid license is required to run Rasa Pro, but all Rasa Pro code is visible to end users and can be customized as needed.

Rasa Pro is the pro-code component of our enterprise solution, Rasa Platform, for implementing resilient and trustworthy AI assistants at scale. Rasa Studio complements Rasa Pro with a low-code user interface, enabling anyone on your team to create and improve your assistant.

---
- 🤓 [Read The Docs](https://rasa.com/docs/rasa-pro/)

- 😁 [Install Rasa Pro](https://rasa.com/docs/rasa-pro/installation/python/installation)

---

## README Contents:
- [Development Internals](#development-internals)
- [Releases](#releases)
- [Troubleshooting](#troubleshooting)

## Development Internals

### Installing Poetry

Rasa uses Poetry for packaging and dependency management. If you want to build it from source,
you have to install Poetry first. Please follow
[the official guide](https://python-poetry.org/docs/#installation) to see all possible options.

To update an existing poetry version to the [version](.github/poetry_version.txt), currently used in rasa, run:
```shell
    poetry self update <version>
```

### Managing environments

The official [Poetry guide](https://python-poetry.org/docs/managing-environments/) suggests to use
[pyenv](https://github.com/pyenv/pyenv) or any other similar tool to easily switch between Python versions.
This is how it can be done:

```bash
pyenv install 3.10.10
pyenv local 3.10.10  # Activate Python 3.10.10 for the current project
```
*Note*: If you have trouble installing a specific version of python on your system
it might be worth trying other supported versions.

By default, Poetry will try to use the currently activated Python version to create the virtual environment
for the current project automatically. You can also create and activate a virtual environment manually — in this
case, Poetry should pick it up and use it to install the dependencies. For example:

```bash
python -m venv .venv
source .venv/bin/activate
```

You can make sure that the environment is picked up by executing

```bash
poetry env info
```

### Building from source

To install dependencies and `rasa` itself in editable mode execute

```bash
make install
```

*Note for macOS users*: under macOS Big Sur we've seen some compiler issues for
dependencies. Using `export SYSTEM_VERSION_COMPAT=1` before the installation helped.


#### Installing optional dependencies

In order to install rasa's optional dependencies, you need to run:

```bash
make install-full
```

*Note for macOS users*: The command `make install-full` could result in a failure while installing `tokenizers`
(issue described in depth [here](https://github.com/huggingface/tokenizers/issues/1050)).

In order to resolve it, you must follow these steps to install a Rust compiler:
```bash
brew install rustup
rustup-init
```

After initialising the Rust compiler, you should restart the console and check its installation:
```bash
rustc --version
```

In case the PATH variable had not been automatically setup, run:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

#### Installing from published Python package

We have a private package registry running at **europe-west3-docker.pkg.dev/rasa-releases/** which hosts python packages as well
as docker containers. To use it, you need to be authenticated.
Follow the steps in the [google documentation](https://cloud.google.com/artifact-registry/docs/python/authentication#keyring)
to make sure `pip` has the necessary credentials to authenticate with the registry.
Afterwards, you should be able to run `pip install rasa`.

To be able to pull the docker image via `docker pull europe-west3-docker.pkg.dev/rasa-releases/rasa/rasa`,
you’ll need to authenticate using the `gcloud auth` command: `gcloud auth configure-docker europe-west3-docker.pkg.dev`.

More information is available in our [public documentation](https://rasa.com/docs/rasa-pro/installation/python/installation).

### Running the Tests

In order to run the tests, make sure that you have set locally the environment variable `RASA_PRO_LICENSE` to a valid license available in 1Password.
You should ensure to install the development requirements:

```bash
make prepare-tests-ubuntu # Only on Ubuntu and Debian based systems
make prepare-tests-macos  # Only on macOS
```

Then, run the tests:

```bash
make test
```

They can also be run at multiple jobs to save some time:

```bash
JOBS=[n] make test
```

Where `[n]` is the number of jobs desired. If omitted, `[n]` will be automatically chosen by pytest.


### Running the Integration Tests

In order to run the integration tests, make sure that you have the development requirements installed:

```bash
make prepare-tests-ubuntu # Only on Ubuntu and Debian based systems
make prepare-tests-macos  # Only on macOS
```

Then, you'll need to start services with the following command which uses
[Docker Compose](https://docs.docker.com/compose/install/):

```bash
make run-integration-containers
```

Finally, you can run the integration tests like this:

```bash
make test-integration
```

In order to run locally the integration tests for the tracing capability, you must first build the rasa image locally.
You can do so using the `docker buildx bake` command.
Note that the rasa image build requires a few base images, which must be built prior to building the rasa image.
The Dockerfiles for these base images are located in the `docker` subdirectory.

You must also set the following environment variables to build the rasa image locally:
- `TARGET_IMAGE_REGISTRY`, e.g. you can either use `rasa` or the private registry `europe-west3-docker.pkg.dev/rasa-releases/rasa-docker`.
- `IMAGE_TAG`, e.g. `localdev`, `latest` or PR ID.
- `BASE_IMAGE_HASH`, e.g. `localdev`
- `BASE_MITIE_IMAGE_HASH`, e.g. `localdev`
- `BASE_BUILDER_IMAGE_HASH`, e.g. `localdev`


### Resolving merge conflicts

Poetry doesn't include any solution that can help to resolve merge conflicts in
the lock file `poetry.lock` by default.
However, there is a great tool called [poetry-merge-lock](https://poetry-merge-lock.readthedocs.io/en/latest/).
Here is how you can install it:

```bash
pip install poetry-merge-lock
```

Just execute this command to resolve merge conflicts in `poetry.lock` automatically:

```bash
poetry-merge-lock
```

### Build a Docker image locally

In order to build a Docker image on your local machine execute the following command:

```bash
make build-docker
```

The Docker image is available on your local machine as `rasa-private-dev`.

### Code Style

To ensure a standardized code style we use the [ruff](https://docs.astral.sh/ruff/formatter/) formatter.
To ensure our type annotations are correct we use the type checker [mypy](https://mypy.readthedocs.io/en/stable/).
If your code is not formatted properly or doesn't type check, GitHub will fail to build.

#### Formatting

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.

If you want to set it up manually, install `ruff` via `poetry install`.
To reformat files execute
```
make formatter
```

#### Type Checking

If you want to check types on the codebase, install `mypy` using `poetry install`.
To check the types execute
```
make types
```

## Releases
Rasa has implemented robust policies governing version naming, as well as release pace for major, minor, and patch releases.

The values for a given version number (MAJOR.MINOR.PATCH) are incremented as follows:
- MAJOR version for incompatible API changes or other breaking changes.
- MINOR version for functionality added in a backward compatible manner.
- PATCH version for backward compatible bug fixes.

The following table describes the version types and their expected *release cadence*:

| Version Type |                                                                  Description                                                                  |  Target Cadence |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| Major        | For significant changes, or when any backward-incompatible changes are introduced to the API or data model.                                   | Every 1 - 2 yrs |
| Minor        | For when new backward-compatible functionality is introduced, a minor feature is introduced, or when a set of smaller features is rolled out. | +/- Quarterly   |
| Patch        | For backward-compatible bug fixes that fix incorrect behavior.                                                                                | As needed       |

While this table represents our target release frequency, we reserve the right to modify it based on changing market conditions and technical requirements.

### Maintenance Policy
Our End of Life policy defines how long a given release is considered supported, as well as how long a release is
considered to be still in active development or maintenance.

The maintenance duration and end of life for every release are shown on our website as part of the [Product Release and Maintenance Policy](https://rasa.com/rasa-product-release-and-maintenance-policy/).

### Cutting a Major / Minor release
#### A week before release day

**Post a message on the engineering Slack channel**, letting the team know you'll be the one cutting the upcoming
release, as well as:
    1. Reminding everyone to go over their issues and PRs and prioritise reviews and merges
    2. Reminding everyone of the scheduled date for the release

#### A day before release day

1. **Evaluate the status of any PR merging that's happening. Follow up with people on their
bugs and fixes.** If the release introduces new bugs or regressions that can't be fixed in time, we should discuss on
Slack about this and take a decision on how to move forward. If the issue is not ready to be merged in time, we remove the issue / PR from the release and notify the PR owner and the product manager on Slack about it. The PR / issue owners are responsible for
communicating any issues which might be release relevant. Postponing the release should be considered as an edge case scenario.

#### Release day! 🚀

1. **At the start of the day, post a small message on slack announcing release day!** Communicate you'll be handling
the release, and the time you're aiming to start releasing (again, no later than 4pm, as issues may arise and
cause delays). This message should be posted early in the morning and before moving forward with any of the steps of the release,
   in order to give enough time to people to check their PRs and issues. That way they can plan any remaining work. A template of the slack message can be found [here](https://rasa-hq.slack.com/archives/C36SS4N8M/p1613032208137500?thread_ts=1612876410.068400&cid=C36SS4N8M).
   The release time should be communicated transparently so that others can plan potentially necessary steps accordingly. If there are bigger changes this should be communicated.
2. Once everything in the release is taken care of, post a small message on Slack communicating you are about to
start the release process (in case anything is missing).
3. **You may now do the release by following the instructions outlined in the
[Rasa Pro README](#steps-to-release-a-new-version) !**

### Steps to release a new version
Releasing a new version is quite simple, as the packages are build and distributed by GitHub Actions.

*Release steps*:
1. Make sure all dependencies are up to date (**especially Rasa SDK**)
    - For Rasa SDK, except in the case of a patch release, that means first creating a [new Rasa SDK release](https://github.com/RasaHQ/rasa-sdk#steps-to-release-a-new-version) (make sure the version numbers between the new Rasa and Rasa SDK releases match)
    - Once the tag with the new Rasa SDK release is pushed and the package appears on [pypi](https://pypi.org/project/rasa-sdk/), the dependency in the rasa repository can be resolved (see below).
2. If this is a minor / major release: Make sure all fixes from currently supported minor versions have been merged from their respective release branches (e.g. 3.8.x) back into main.
3. In case of a minor release, create a new branch that corresponds to the new release, e.g.
   ```bash
    git checkout -b 3.8.x
    git push origin 3.8.x
    ```
4. Switch to the branch you want to cut the release from (`main` in case of a major, the `<major>.<minor>.x` branch for minors and patches)
    - Update the `rasa-sdk` entry in `pyproject.toml` with the new release version and run `poetry update`. This creates a new `poetry.lock` file with all dependencies resolved.
    - Commit the changes with `git commit -am "bump rasa-sdk dependency"` but do not push them. They will be automatically picked up by the following step.
5. Run `make release`
6. Create a PR against the release branch (e.g. `3.8.x`)
7. Once your PR is merged, [this](https://github.com/RasaHQ/rasa-private/actions/workflows/tag-release.yml) workflow runs and an automatic tag is created and pushed to remote.
   (If this fails for some reason, then run the following manually on the release branch) :
    ```bash
    git checkout 3.8.x
    git pull origin 3.8.x
    git tag 3.8.0 -m "next release"
    git push origin 3.8.0 --tags
    ```
    GitHub will build this tag and publish the build artifacts.
8. After all the steps are completed and if everything goes well then we should see a message automatically posted in the company's Slack (`release` channel) like this [one](https://rasa-hq.slack.com/archives/C7B08Q5FX/p1614354499046600)
9. If however an error occurs in the build, then we should see a failure message automatically posted in the company's Slack (`dev-tribe` channel) like this [one](https://rasa-hq.slack.com/archives/C01M5TAHDHA/p1701444735622919)
   (In this case do the following checks):
    - Check the workflows in [Github Actions](https://github.com/RasaHQ/rasa-private/actions) and make sure that the merged PR of the current release is completed successfully. To easily find your PR you can use the filters `event: push` and `branch: <version number>` (example on release 2.4 you can see [here](https://github.com/RasaHQ/rasa/actions/runs/643344876))
    - If the workflow is not completed, then try to re run the workflow in case that solves the problem
    - If the problem persists, check also the log files and try to find the root cause of the issue
    - If you still cannot resolve the error, contact the infrastructure team by providing any helpful information from your investigation

### Cutting a Patch release

Patch releases are simpler to cut, since they are meant to contain only bugfixes.

**The only things you need to do to cut a patch release are:**

1. Notify the engineering team on Slack that you are planning to cut a patch, in case someone has an important fix
to add.
2. Make sure the bugfix(es) are in the release branch you will use (p.e if you are cutting a `3.8.2` patch, you will
need your fixes to be on the `3.8.x` release branch). All patch releases must come from a `.x` branch!
3. Once you're ready to release the Rasa Pro patch, checkout the branch, run `make release` and follow the
steps + get the PR merged.
4. Once the PR is in, wait for the [tag release workflow](https://github.com/RasaHQ/rasa-private/actions/workflows/tag-release.yml) to create the tag.
   (If this fails for some reason, then run the following manually on the release branch) :
    ```bash
    git checkout 3.8.x
    git pull origin 3.8.x
    git tag 3.8.0 -m "next release"
    git push origin 3.8.0 --tags
    ```
5. After this you should see the CI workflow "Continuous Integration" in the Actions tab with the relevant tag name. Keep an eye on it to make sure it is successful as sometimes retries might be required. 
6. After all the steps are completed and if everything goes well then we should see a message automatically posted in the company's Slack (`release` channel) like this [one](https://rasa-hq.slack.com/archives/C7B08Q5FX/p1614354499046600)
7. If however an error occurs in the build, then we should see a failure message automatically posted in the company's Slack (`dev-tribe` channel) like this [one](https://rasa-hq.slack.com/archives/C01M5TAHDHA/p1701444735622919)

Make sure to merge the branch `3.7.x` after your PR with `main`. This needs to be done manually until Roberto is added (see [ATO-2091](https://rasahq.atlassian.net/browse/ATO-2091))

### Cutting a Pre release version

A Pre release version is an alpha, beta, dev or rc version. For more details on which version you require refer to the [Rasa Software Release Lifecycle](https://www.notion.so/rasa/Rasa-Software-Release-Lifecycle-eb704d75f87646a9a9aca1f3fbe71fb3#6e26ac9a15b64f91bb94d6bfea9306a0)

1. Make sure you are using the right branch for the release, for instance pre releases are always made from either the main or a feature branch (especially for a dev release)
2. Once you're ready to release, checkout the branch, run `make release` and follow the
steps.
3. Only in case of a pre release, the release branch created will be prefixed with 'prepare-release-pre-'
4. Note that when releasing from a feature branch the 'prepare-release-pre' branch will not be created automatically and has to be done manually. This is done to ensure all major/minor/patch releases only happens from the correct branches.
   (In this case the version updates will be added to the same branch as a commit, and you will have to manually create a `prepare-release-pre-' branch and push to remote)
5. Only in case of a pre release, we currently skip all test runs and docker image builds on a 'prepare-release-pre-' PR. This was done to speed up the pre release process.
6. Once your PR gets merged, the [tag release workflow](https://github.com/RasaHQ/rasa-private/actions/workflows/tag-release.yml) will create the tag.
7. After this you should see the CI workflow "Continuous Integration" in the Actions tab with the relevant tag name. Keep an eye on it to make sure it is successful as sometimes retries might be required. 
8. After all the steps are completed and if everything goes well then we should see a message automatically posted in the company's Slack (`release` channel) like this [one](https://rasa-hq.slack.com/archives/C7B08Q5FX/p1614354499046600)
9. If however an error occurs in the build, then we should see a failure message automatically posted in the company's Slack (`dev-tribe` channel) like this [one](https://rasa-hq.slack.com/archives/C01M5TAHDHA/p1701444735622919)


### Actively maintained versions

Please refer to the [Rasa Product Release and Maintenance Policy](https://rasa.com/rasa-product-release-and-maintenance-policy/) page.

### Active workflows on the CI

Please refer to the [WORKFLOW_README FILE](https://github.com/RasaHQ/rasa-private/blob/main/WORKFLOW_README.md)

## Troubleshooting

- When running docker commands, if you encounter this error: `OSError No space left on device`, consider running:

  ```shell
  docker system prune --all
  ```

  For more information on this command, please see the [Official Docker Documentation](https://docs.docker.com/engine/reference/commandline/system_prune/).
