# Rasa Open Source

[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/rasa.svg)](https://badge.fury.io/py/rasa)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa.svg)](https://pypi.python.org/pypi/rasa)
[![Build Status](https://github.com/RasaHQ/rasa/workflows/Continuous%20Integration/badge.svg)](https://github.com/RasaHQ/rasa/actions)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa/badge.svg?branch=master)](https://coveralls.io/github/RasaHQ/rasa?branch=master)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)
![Documentation Build](https://img.shields.io/netlify/d2e447e4-5a5e-4dc7-be5d-7c04ae7ff706?label=Documentation%20Build)
[![FOSSA Status](https://app.fossa.com/api/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git.svg?type=shield)](https://app.fossa.com/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git?ref=badge_shield)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/orgs/RasaHQ/projects/23)

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png">

Rasa is an open source machine learning framework to automate text-and voice-based conversations. With Rasa, you can build contextual assistants on:
- Facebook Messenger
- Slack
- Google Hangouts
- Webex Teams
- Microsoft Bot Framework
- Rocket.Chat
- Mattermost
- Telegram
- Twilio
- Your own custom conversational channels

or voice assistants as:
- Alexa Skills
- Google Home Actions

Rasa helps you build contextual assistants capable of having layered conversations with 
lots of back-and-forth. In order for a human to have a meaningful exchange with a contextual 
assistant, the assistant needs to be able to use context to build on things that were previously 
discussed – Rasa enables you to build assistants that can do this in a scalable way.

There's a lot more background information in this
[blog post](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a).

---
- **What does Rasa do? 🤔**
  [Check out our Website](https://rasa.com/)

- **I'm new to Rasa 😄**
  [Get Started with Rasa](https://rasa.com/docs/getting-started/)

- **I'd like to read the detailed docs 🤓**
  [Read The Docs](https://rasa.com/docs/)

- **I'm ready to install Rasa 🚀**
  [Installation](https://rasa.com/docs/rasa/user-guide/installation/)

- **I want to learn how to use Rasa 🚀**
  [Tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/)

- **I have a question ❓**
  [Rasa Community Forum](https://forum.rasa.com/)

- **I would like to contribute 🤗**
  [How to Contribute](#how-to-contribute)

---  
## Where to get help

There is extensive documentation in the [Rasa Docs](https://rasa.com/docs/rasa).
Make sure to select the correct version so you are looking at
the docs for the version you installed.

Please use [Rasa Community Forum](https://forum.rasa.com) for quick answers to
questions.

### README Contents:
- [How to contribute](#how-to-contribute)
- [Development Internals](#development-internals)
- [License](#license)

### How to contribute
We are very happy to receive and merge your contributions into this repository! 

To contribute via pull request, follow these steps:

1. Create an issue describing the feature you want to work on (or
   have a look at the [contributor board](https://github.com/orgs/RasaHQ/projects/23))
2. Write your code, tests and documentation, and format them with ``black``
3. Create a pull request describing your changes

For more detailed instructions on how to contribute code, check out these [code contributor guidelines](CONTRIBUTING.md).

You can find more information about how to contribute to Rasa (in lots of
different ways!) [on our website.](http://rasa.com/community/contribute).

Your pull request will be reviewed by a maintainer, who will get
back to you about any necessary changes or questions. You will
also be asked to sign a
[Contributor License Agreement](https://cla-assistant.io/RasaHQ/rasa).


## Development Internals

### Installing Poetry

Rasa uses Poetry for packaging and dependency management. If you want to build it from source,
you have to install Poetry first. This is how it can be done:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

There are several other ways to install Poetry. Please, follow 
[the official guide](https://python-poetry.org/docs/#installation) to see all possible options.

### Managing environments

The official [Poetry guide](https://python-poetry.org/docs/managing-environments/) suggests to use
[pyenv](https://github.com/pyenv/pyenv) or any other similar tool to easily switch between Python versions. 
This is how it can be done:

```bash
pyenv install 3.7.6
pyenv local 3.7.6  # Activate Python 3.7.6 for the current project
```

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

### Running and changing the documentation

First of all, install all the required dependencies:

```bash
make install install-docs
```

After the installation has finished, you can run and view the documentation
locally using:

```bash
make livedocs
```

It should open a new tab with the local version of the docs in your browser;
if not, visit http://localhost:3000 in your browser.
You can now change the docs locally and the web page will automatically reload
and apply your changes.

### Running the Tests

In order to run the tests, make sure that you have the development requirements installed:

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

### Code Style

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black).
To ensure our type annotations are correct we use the type checker [pytype](https://github.com/google/pytype). 
If your code is not formatted properly or doesn't type check, GitHub will fail to build.

#### Formatting

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.

If you want to set it up manually, install black via `poetry install`.
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

### Deploying documentation updates

We use `Docusaurus v2` to build docs for tagged versions and for the master branch.
The static site that gets built is pushed to the `documentation` branch of this repo.

We host the site on netlify. On master branch builds (see `.github/workflows/documentation.yml`), we push the built docs to
the `documentation` branch. Netlify automatically re-deploys the docs pages whenever there is a change to that branch.

## Releases
### Release Timeline for Minor Releases
**For Rasa Open Source, we usually commit to time-based releases, specifically on a monthly basis.** 
This means that we commit beforehand to releasing a specific version of Rasa Open Source on a specific day, 
and we cannot be 100% sure what will go in a release, because certain features may not be ready. 

At the beginning of each quarter, the Rasa team will review the scheduled release dates for all products and make sure 
they work for the projected work we have planned for the quarter, as well as work well across products.

**Once the dates are settled upon, we update the respective [milestones](https://github.com/RasaHQ/rasa/milestones).**

### Cutting a Major / Minor release
#### A week before release day

1. **Make sure the [milestone](https://github.com/RasaHQ/rasa/milestones) already exists and is scheduled for the 
correct date.**
2. **Take a look at the issues & PRs that are in the milestone**: does it look about right for the release highlights 
we are planning to ship? Does it look like anything is missing? Don't worry about being aware of every PR that should 
be in, but it's useful to take a moment to evaluate what's assigned to the milestone.
3. **Post a message on the engineering Slack channel**, letting the team know you'll be the one cutting the upcoming 
release, as well as:
    1. Providing the link to the appropriate milestone
    2. Reminding everyone to go over their issues and PRs and please assign them to the milestone
    3. Reminding everyone of the scheduled date for the release

#### A day before release day

1. **Go over the milestone and evaluate the status of any PR merging that's happening. Follow up with people on their 
bugs and fixes.** If the release introduces new bugs or regressions that can't be fixed in time, we should discuss on 
Slack about this and take a decision to go forward or postpone the release. The PR / issue owners are responsible for 
communicating any issues which might be release relevant.

#### Release day! 🚀

1. **At the start of the day, post a small message on slack announcing release day!**. Communicate you'll be handling 
the release, and the time you're aiming to start releasing (again, no later than 4pm, as issues may arise and 
cause delays)
2. Make sure the milestone is empty (everything has been either merged or moved to the next milestone)
3. Once everything in the milestone is taken care of, post a small message on Slack communicating you are about to 
start the release process (in case anything is missing).
4. **You may now do the release by following the instructions outlined in the 
[Rasa Open Source README](#steps-to-release-a-new-version) !**

### Steps to release a new version
Releasing a new version is quite simple, as the packages are build and distributed by GitHub Actions.

*Terminology*:
* micro release (third version part increases): 1.1.2 -> 1.1.3
* minor release (second version part increases): 1.1.3 -> 1.2.0
* major release (first version part increases): 1.2.0 -> 2.0.0

*Release steps*:
1. Make sure all dependencies are up to date (**especially Rasa SDK**)
    - For Rasa SDK that means first creating a [new Rasa SDK release](https://github.com/RasaHQ/rasa-sdk#steps-to-release-a-new-version) (make sure the version numbers between the new Rasa and Rasa SDK releases match)
    - Once the tag with the new Rasa SDK release is pushed and the package appears on [pypi](https://pypi.org/project/rasa-sdk/), the dependency in the rasa repository can be resolved (see below).
2. Switch to the branch you want to cut the release from (`master` in case of a major / minor, the current feature branch for micro releases) 
    - Update the `rasa-sdk` entry in `pyproject.toml` with the new release version and run `poetry update`. This creates a new `poetry.lock` file with all dependencies resolved.
    - Commit the changes with `git commit -am "bump rasa-sdk dependency"` but do not push them. They will be automatically picked up by the following step.
3. Run `make release`
4. Create a PR against master or the release branch (e.g. `1.2.x`)
5. Once your PR is merged, tag a new release (this SHOULD always happen on master or release branches), e.g. using
    ```bash
    git tag 1.2.0 -m "next release"
    git push origin 1.2.0
    ```
    GitHub will build this tag and publish the build artifacts.
6. **If this is a minor release**, a new release branch should be created pointing to the same commit as the tag to allow for future patch releases, e.g.
    ```bash
    git checkout -b 1.2.x
    git push origin 1.2.x
    ```

### Cutting a Micro release

Micro releases are simpler to cut, since they are meant to contain only bugfixes.

**The only things you need to do to cut a micro are:**

1. Notify the engineering team on Slack that you are planning to cut a micro, in case someone has an important fix 
to add.
2. Make sure the bugfix(es) are in the release branch you will use (p.e if you are cutting a `2.0.4` micro, you will 
need your fixes to be on the `2.0.x` release branch). All micros must come from a `.x` branch!
3. Once you're ready to release the Rasa Open Source micro, checkout the branch, run `make release` and follow the 
steps + get the PR merged.
4. Once the PR is in, pull the `.x` branch again and push the tag!

## License
Licensed under the Apache License, Version 2.0.
Copyright 2020 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

A list of the Licenses of the dependencies of the project can be found at
the bottom of the
[Libraries Summary](https://libraries.io/github/RasaHQ/rasa).
