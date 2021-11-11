<h1 align="center">Rasa Open Source</h1>

<div align="center">

[![Join the chat on Rasa Community Forum](https://img.shields.io/badge/forum-join%20discussions-brightgreen.svg)](https://forum.rasa.com/?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/rasa.svg)](https://badge.fury.io/py/rasa)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa.svg)](https://pypi.python.org/pypi/rasa)
[![Build Status](https://github.com/RasaHQ/rasa/workflows/Continuous%20Integration/badge.svg)](https://github.com/RasaHQ/rasa/actions)
[![Coverage Status](https://coveralls.io/repos/github/RasaHQ/rasa/badge.svg?branch=main)](https://coveralls.io/github/RasaHQ/rasa?branch=main)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)
![Documentation Build](https://img.shields.io/netlify/d2e447e4-5a5e-4dc7-be5d-7c04ae7ff706?label=Documentation%20Build)
[![FOSSA Status](https://app.fossa.com/api/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git.svg?type=shield)](https://app.fossa.com/projects/custom%2B8141%2Fgit%40github.com%3ARasaHQ%2Frasa.git?ref=badge_shield)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/orgs/RasaHQ/projects/23)

</div>

<a href="https://grnh.se/05a908c02us" target="_blank"><img align="center" src="https://www.rasa.com/assets/img/github/hiring_banner.png" alt="An image with Sara, the Rasa mascot, standing next to a roadmap with future Rasa milestones: identifying unsuccessful conversations at scale, continuous model evaluation, controllable NLG and breaking free from intents. Are you excited about these milestones? Help us make these ideas become reality - we're hiring!" title="We're hiring! Learn more"></a>

<hr />

💡 **Rasa 오픈 소스 3.0이 출시됩니다!** 💡


3.0의 아키텍처 변경 작업을 수행하는 동안 마이너 릴리즈를 잠시 중단해야 하기 때문에 [2.8](https://github.com/RasaHQ/rasa/milestone/39)은 2.x 시리즈의 마지막 마이너 버전이 될 것입니다. 당신은 3.0과 함께 출시할 예정인 새로운 기능과 개선 사항에 계속 기여할 수 있습니다. [컨트리뷰터 가이드라인](#how-to-contribute)에 대해서 자세히 알아보세요.

빠른 피드백을 받기 위해 앞으로 몇 달 동안 알파 릴리즈와 릴리즈 후보를 출시할 계획입니다. 계속 지켜봐 주세요!
<hr />

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png" alt="An image of Sara, the Rasa mascot bird, holding a flag that reads Open Source with one wing, and a wrench in the other" title="Rasa Open Source">
Rasa는 텍스트 및 음성 기반 대화를 자동화하는 오픈소스 머신러닝 프레임워크입니다. Rasa를 사용하면 다음과 같은 상황별 기능을 구축할 수 있습니다:<br/><br/>
- 페이스북 메신저(Facebook Messenger)<br />
- 슬랙(Slack)<br />
- 구글 행아웃(Google Hangouts)<br />
- 웹엑스 팀즈 (Webex Teams)<br />
- 마이크로소프트 봇 프레임워크(Microsoft Bot Framework)<br />
- 로켓챗(Rocket.chat)<br />
- 매터모스트(Mattermost)<br />
- 텔레그램(Telegram)<br />
- 트윌리오(Twilio)<br />
- 나만의 맞춤 대화 채널<br /><br />


또는 다음과 같은 음성 비서를 구축할 수 있습니다:
- 알렉사 스킬(Alexa Skills)
- 구글 홈 액션(Google Home Actions)

Rasa는 많은 대화를 주고받을 수 있는 상황별 어시스턴트를 구축하는 데 도움이 됩니다.사람이 상황에 따라 비서와 의미 있는 교환을 하려면 비서가 상황을 이용하여 이전에 논의된 내용을 구축할 수 있어야 합니다. Rasa를 사용하면 확장 가능한 방식으로 이를 수행할 수 있는 비서를 구축할 수 있습니다.

이 [블로그 게시물](https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a)에는 더 많은 배경 정보를 확인할 수 있습니다. 


---
- **Rasa는 무엇을 하나요? 🤔**
  [우리 웹사이트를 확인하세요](https://rasa.com/)

- **나는 Rasa를 처음 사용합니다 😄**
  [Rasa 시작하기](https://rasa.com/docs/getting-started/)

- **자세한 문서를 읽어보고 싶습니다 🤓**
  [문서 읽어보기](https://rasa.com/docs/)

- **Rasa를 설치할 준비가 되었습니다 🚀**
  [설치](https://rasa.com/docs/rasa/user-guide/installation/)

- **Rasa 사용법을 배우고 싶어요 🚀**
  [튜토리얼](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/)

- **질문이 있어요 ❓**
  [Rasa 커뮤니티 포럼](https://forum.rasa.com/)

- **기여하고 싶어요 🤗**
  [기여 방법](#how-to-contribute)

---
## 도움을 받을 수 있는 곳

[Rasa Docs](https://rasa.com/docs/rasa).에는 광범위한 문서들이 있습니다. 설치한 버전에 대한 문서를 볼 수 있도록 올바른 버전을 선택했는지 확인하세요.

질문에 대한 빠른 답변은 [Rasa 커뮤니티 포럼](https://forum.rasa.com)을 이용해주세요.

### README 내용:
- [기여하는 방법](#how-to-contribute)
- [개발 내부 내용](#development-internals)
- [출시](#releases)
- [라이센스](#license)

### how-to-contribute
우리는 귀하의 기여를 이 레포지토리에 merge할 수 있게 되어 매우 기쁩니다!

pull 요청을 통해 기여하려면 다음 단계를 따르세요:


1. 작업하려는 기능을 설명하는 issue를 만듭니다 (또는
    [컨트리뷰터 보드](https://github.com/orgs/RasaHQ/projects/23)을 참조하세요.)
2. 코드, 테스트 및 문서를 작성하고 ``black``으로 형식을 지정합니다
3. 변경 사항을 설명하는 pull request를 생성합니다

코드를 기여하는 방법에 대한 자세한 지침은 [코드 컨트리뷰터 가이드라인](CONTRIBUTING.md)을 확인하세요.

[저희 웹사이트](http://rasa.com/community/contribute)에서 Rasa에 기여하는 방법에 대한 자세한 정보를 찾을 수 있습니다. (
다른 많은 방법들도 가능합니다!) 

귀하의 풀 리퀘스트에 대한 검토는 유지 보수 담당자가 진행하며, 담당자가 필요한 변경 사항이나 질문에 대해 회신해 드릴 것입니다. 
또한 [컨트리뷰터 라이센스 계약](https://cla-assistant.io/RasaHQ/rasa)에 귀하의 서명을 요청할 것입니다.


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
pyenv install 3.7.9
pyenv local 3.7.9  # Activate Python 3.7.9 for the current project
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

The Docker image is available on your local machine as `rasa:localdev`.

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

We use `Docusaurus v2` to build docs for tagged versions and for the `main` branch.
The static site that gets built is pushed to the `documentation` branch of this repo.

We host the site on netlify. On `main` branch builds (see `.github/workflows/documentation.yml`), we push the built docs to
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
Slack about this and take a decision on how to move forward. If the issue is not ready to be merged in time, we remove the issue / PR from the milestone and notify the PR owner and the product manager on Slack about it. The PR / issue owners are responsible for
communicating any issues which might be release relevant. Postponing the release should be considered as an edge case scenario.

#### Release day! 🚀

1. **At the start of the day, post a small message on slack announcing release day!** Communicate you'll be handling
the release, and the time you're aiming to start releasing (again, no later than 4pm, as issues may arise and
cause delays). This message should be posted early in the morning and before moving forward with any of the steps of the release, 
   in order to give enough time to people to check their PRs and issues. That way they can plan any remaining work. A template of the slack message can be found [here](https://rasa-hq.slack.com/archives/C36SS4N8M/p1613032208137500?thread_ts=1612876410.068400&cid=C36SS4N8M).
   The release time should be communicated transparently so that others can plan potentially necessary steps accordingly. If there are bigger changes this should be communicated.
2. Make sure the milestone is empty (everything has been either merged or moved to the next milestone)
3. Once everything in the milestone is taken care of, post a small message on Slack communicating you are about to
start the release process (in case anything is missing).
4. **You may now do the release by following the instructions outlined in the
[Rasa Open Source README](#steps-to-release-a-new-version) !**

#### After a Major release

After a Major release has been completed, please follow [these instructions to complete the documentation update](./docs/README.md#manual-steps-after-a-new-version).

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
2. In case of a minor release, create a new branch that corresponds to the new release, e.g. 
   ```bash
    git checkout -b 1.2.x
    git push origin 1.2.x
    ```
3. Switch to the branch you want to cut the release from (`main` in case of a major, the `<major>.<minor>.x` branch for minors and micros)
    - Update the `rasa-sdk` entry in `pyproject.toml` with the new release version and run `poetry update`. This creates a new `poetry.lock` file with all dependencies resolved.
    - Commit the changes with `git commit -am "bump rasa-sdk dependency"` but do not push them. They will be automatically picked up by the following step.
4. If this is a major release, update the list of actively maintained versions [in the README](#actively-maintained-versions) and in [the docs](./docs/docs/actively-maintained-versions.mdx).
5. Run `make release`
6. Create a PR against the release branch (e.g. `1.2.x`)
7. Once your PR is merged, tag a new release (this SHOULD always happen on the release branch), e.g. using
    ```bash
    git checkout 1.2.x
    git pull origin 1.2.x
    git tag 1.2.0 -m "next release"
    git push origin 1.2.0
    ```
    GitHub will build this tag and publish the build artifacts.
8. After all the steps are completed and if everything goes well then we should see a message automatically posted in the company's Slack (`product` channel) like this [one](https://rasa-hq.slack.com/archives/C7B08Q5FX/p1614354499046600)
9. If no message appears in the channel then you can do the following checks:
    - Check the workflows in [Github Actions](https://github.com/RasaHQ/rasa/actions) and make sure that the merged PR of the current release is completed successfully. To easily find your PR you can use the filters `event: push` and `branch: <version number>` (example on release 2.4 you can see [here](https://github.com/RasaHQ/rasa/actions/runs/643344876))
    - If the workflow is not completed, then try to re run the workflow in case that solves the problem
    - If the problem persists, check also the log files and try to find the root cause of the issue
    - If you still cannot resolve the error, contact the infrastructure team by providing any helpful information from your investigation
10.  After the message is posted correctly in the `product` channel, check also in the `product-engineering-alerts` channel if there are any alerts related to the Rasa Open Source release like this [one](https://rasa-hq.slack.com/archives/C01585AN2NP/p1615486087001000)
    
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

### Actively maintained versions

We're actively maintaining _any minor on our latest major release_ and _the latest minor of the previous major release_.
Currently, this means the following minor versions will receive bugfixes updates:
- 1.10
- Every minor version on 2.x

## License
Licensed under the Apache License, Version 2.0.
Copyright 2021 Rasa Technologies GmbH. [Copy of the license](LICENSE.txt).

A list of the Licenses of the dependencies of the project can be found at
the bottom of the
[Libraries Summary](https://libraries.io/github/RasaHQ/rasa).
