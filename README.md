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

💡 **Rasa Open Source 3.0 is coming up!** 💡

[2.8](https://github.com/RasaHQ/rasa/milestone/39) will be the last minor in the 2.x series, as we need to pause releasing minors while we work on architectural changes in 3.0. You can still contribute new features and improvements which we plan to release together with 3.0. Read more
about [our contributor guidelines](#how-to-contribute).

We plan to ship alpha releases and release candidates over the next few months in order to get early feedback. Stay tuned!

<hr />

<img align="right" height="244" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png" alt="An image of Sara, the Rasa mascot bird, holding a flag that reads Open Source with one wing, and a wrench in the other" title="Rasa Open Source">

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
- [Releases](#releases)
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

### Poetry 설치

Rasa는 패키징과 의존성 관리를 위해 Poetry를 사용합니다. 원본에서 빌드하고 싶다면, 먼저 Poetry를 설치해야 합니다. 설치 방법:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Poetry를 설치하는 몇 가지 다른 방법도 있습니다. 가능한 모든 옵션을 보려면 [공식 가이드](https://python-poetry.org/docs/#installation)를 확인하십시오.

### 환경 관리

공식 [Poetry 가이드](https://python-poetry.org/docs/managing-environments/)에서는 파이썬 버전 간에 쉽게 전환할 수 있도록 [pyenv](https://github.com/pyenv/pyenv) 또는 다른 비슷한 도구를 사용할 것을 제안합니다. 설치 방법: 

```bash
pyenv install 3.7.9
pyenv local 3.7.9  # 현재 프로젝트에 대해 파이썬 3.7.9 활성화
```
*주의*: 특정 버전의 파이썬을 설치하는 데 문제가 있는 경우 지원되는 다른 버전을 사용하십시오.

기본적으로, Poetry는 현재 활성화된 파이썬 버전을 사용하여 현재 프로젝트의 가상 환경을 자동으로 생성하려고 시도 할 것입니다. 가상 환경을 수동으로 만들고 활성화할 수도 있습니다. — 이 경우, Poetry는 그것을 dependencies를 설치하는데 사용해야 합니다. 예를 들어:

```bash
python -m venv .venv
source .venv/bin/activate
```

실행을 통해 환경이 선택되었는지 확인할 수 있습니다

```bash
poetry env info
```

### 원본에서 빌드

편집 가능한 모드에서 dependencies와 `rasa`를 설치하려면

```bash
make install
```

*macOS 유저라면*: macOS Big Sur에 dependencies에 대한 몇가지 컴파일러 문제가 있습니다.
. 설치 전에 `export SYSTEM_VERSION_COMPAT=1` 사용하면 도움이 될 것입니다.

### documentation 실행 및 변경

먼저, 필요한 모든 dependencies를 설치하십시오:

```bash
make install install-docs
```

설치가 완료되면 아래 코드를 사용하여 문서를 실행하고 볼 수 있습니다.

```bash
make livedocs
```

브라우저에 있는 문서의 로컬 버전으로 새 탭을 열여야 합니다;
열지 못했다면, 브라우저에서 http://localhost:3000 에 접속하십시오.
이제 문서를 로컬에서 변경할 수 있으며 웹 페이지가 자동으로 로드되어 변경 내용을 적용합니다.

### 테스트 실행

테스트를 실행하려면, 먼저 개발 요구 사항이 설치되어 있는지 확인하십시오:

```bash
make prepare-tests-ubuntu # Ubuntu, Debian based systems에서만
make prepare-tests-macos  # macOS에서만
```

그리고, 테스트를 실행하세요:

```bash
make test
```

시간을 절약하기 위해 여러 작업을 실행할 수 있습니다:

```bash
JOBS=[n] make test
```

`[n]`은 원하는 작업의 개수입니다. 생략할 경우, `[n]`은 pytest를 통해 자동으로 선택됩니다.


### 통합 테스트 실행

통합 테스트를 실행하려면, 개발 요구 사항이 설치되어 있는지 확인해야 합니다:

```bash
make prepare-tests-ubuntu # Ubuntu, Debian based systems에서만
make prepare-tests-macos  # macOS에서만
```

그런 다음, [Docker Compose](https://docs.docker.com/compose/install/)를 사용하는 다음 명령으로 서비스를 시작해야 합니다:

```bash
make run-integration-containers
```

마지막으로, 다음과 같은 통합 테스트를 실행할 수 있습니다:

```bash
make test-integration
```


### 병합 충돌 해결

Poetry에는 기본적으로 잠금 파일 `poetry.lock`의 병합 충돌을 해결하는 데 도움이 되는 솔루션이 포함되어 있지 않습니다.
그러나, [poetry-merge-lock](https://poetry-merge-lock.readthedocs.io/en/latest/)라는 좋은 도구가 있습니다.
설치 방법입니다:

```bash
pip install poetry-merge-lock
```

`poetry.lock`에서 병합 충돌을 자동으로 해결하려면 이 명령을 실행하십시오:

```bash
poetry-merge-lock
```

### 도커 이미지 로컬 작성

로컬 컴퓨터에 도커 이미지를 작성하려면 다음 명령을 실행하십시오:

```bash
make build-docker
```

도커 이미지는 로컬 컴퓨터에서 `rasa:localdev`로 사용할 수 있습니다.

### 코드 스타일

표준화된 코드 스타일을 위해 포맷터 [black](https://github.com/ambv/black)을 사용합니다.
유형 주석이 올바른지 확인하기 위해 [pytype](https://github.com/google/pytype)을 사용합니다.
당신의 코드가 제대로 포맷되지 않았거나 Check되지 않았다면, GitHub가 빌드할 수 없습니다.

#### 서식 설정

모든 커밋에서 코드를 자동으로 포맷하려면 [pre-commit](https://pre-commit.com/)을 사용하십시오.
`pip install pre-commit`을 통해 설치하고 루트 폴더에서 `pre-commit install`을 실행하면 됩니다.
이렇게 하면 모든 커밋에서 파일을 재구성하는 후크가 저장소에 추가됩니다.

수동으로 설정하려면 `poetry install`을 통해 black을 설치하십시오.
파일을 다시 포맷하려면 아래 코드를 실행하십시오.
```
make formatter
```

#### 유형 확인

코드베이스에서 타입을 확인하려면 `poetry install`을 사용하여 `mypy`을 설치하십시오.
타입을 확인하려면 아래 코드를 실행하십시오.
```
make types
```

### 문서 업데이트 배포

우리는 `Docusaurus v2`를 사용하여 태그가 지정된 버전과 `main` 브랜치에 대한 문서를 작성합니다.
빌드되는 정적 사이트는 이 저장소의 `documentation` 브랜치로 푸시됩니다.

우리는 netlify로 사이트를 주최합니다. `main` 브랜치 빌드에서 (`.github/workflows/documentation.yml`를 확인하세요), 우리는 빌드된 문서를 `documentation` 브랜치로 푸시합니다. Netlify는 해당 브랜치가 변경될 때마다 자동으로 문서 페이지를 다시 배포합니다

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
