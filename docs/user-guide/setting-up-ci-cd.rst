:desc: Set up a CI/CD pipeline to ensure that iterative improvements to your assistant are tested
       and deployed with minimum manual effort

.. _setting-up-ci-cd:

Setting up CI/CD
================

Even though developing a contextual assistant is different from developing traditional
software, you should still follow software development best practices.
Setting up a Continuous Integration (CI) and Continuous Deployment (CD)
pipeline ensures that incremental updates to your bot are improving it, not harming it.

.. contents::
   :local:
   :depth: 2


Overview
--------

Continous Integration (CI) is the practice of merging in code changes
frequently and automatically testing changes as they are committed. Continuous
Deployment (CD) means automatically deploying integrated changes to a staging
or production environment. Together, they allow you to make more frequent improvements
to your assistant and efficiently test and deploy those changes.

This guide will cover **what** should go in a CI/CD pipeline, specific to a
Rasa project. **How** you implement that pipeline is up to you.
There are many CI/CD tools out there, such as `GitHub Actions <https://github.com/features/actions>`_,
`GitLab CI/CD <https://docs.gitlab.com/ee/ci/>`_, `Jenkins <https://www.jenkins.io/doc/>`_, and
`CircleCI <https://circleci.com/docs/2.0/>`_. We recommend choosing a tool that integrates with
whatever Git repository you use.


Continuous Integration (CI)
---------------------------

The best way to improve an assistant is with frequent `incremental updates
<https://rasa.com/docs/rasa-x/user-guide/fix-problems>`_.
No matter how small a change is, you want to be sure that it doesn't introduce
new problems or negatively impact the performance of your assistant.

It is usually best to run CI checks on merge / pull requests or on commit. Most tests are
quick enough to run on every change. However, you can choose to run more
resource-intensive tests only when certain files have been changed or when some
other indicator is present. For example, if your code is hosted on Github,
you can make a test run only if the pull request has a certain label (e.g. "NLU testing required").

.. contents::
   :local:

Validate Data and Stories
#########################

:ref:`Data validation <validate-files>` verifies that there are no mistakes or
major inconsistencies in your domain file, NLU data, or story data.

.. code-block:: bash

   rasa data validate --fail-on-warnings --max-history <max_history>

If data validation results in errors, training a model will also fail. By
including the ``--fail-on-warnings`` flag, validation will also fail on
warnings about problems that won't prevent training a model, but might indicate
messy data, such as actions listed in the domain that aren't used in any
stories.

Data validation includes :ref:`story structure validation <test-story-files-for-conflicts>`.
Story validation checks if you have any
stories where different bot actions follow from the same dialogue history.
Conflicts between stories will prevent a model from learning the correct
pattern for a dialogue. Set the ``--max-history`` parameter to the value of ``max_history`` for the
memoization policy in your ``config.yml``. If you haven't set one, use the default of ``5``.

Train a Model
#############

.. code-block:: bash

   rasa train

Training a model verifies that your NLU pipeline and policy configurations are
valid and trainable, and it provides a model to use for test conversations.
If it passes the CI tests, then you can also :ref:`upload the trained model <uploading-a-model>`
to your server as part of the continuous deployment process .

.. _test-the-assistant:

Test the Assistant
##################

Testing your trained model on :ref:`test conversations
<end-to-end-testing>` is the best way to have confidence in how your assistant
will act in certain situations. These stories, written in a modified story
format, allow you to provide entire conversations and test that, given this
user input, your model will behave in the expected manner. This is especially
important as you start introducing more complicated stories from user
conversations.


.. code-block:: bash

   rasa test --stories tests/conversation_tests.md --fail-on-prediction-errors

The ``--fail-on-prediction-errors`` flag ensures the test will fail if any test
conversation fails.

End-to-end testing is only as thorough and accurate as the test
cases you include, so you should continue to grow your set of test conversations
as you make improvements to your assistant. A good rule of thumb to follow is that you should aim for your test conversations
to be representative of the true distribution of real conversations.
Rasa X makes it easy to `add test conversations based on real conversations <https://rasa.com/docs/rasa-x/user-guide/test-assistant/#how-to-create-tests>`_.

Note: End-to-end testing does **not** execute your action code. You will need to
:ref:`test your action code <testing-action-code>` in a seperate step.

Compare NLU Performance
#######################

If you've made significant changes to your NLU training data (e.g.
splitting an intent into two intents or adding a lot of training examples), you should run a
:ref:`full NLU evaluation <nlu-evaluation>`. You'll want to compare
the performance of the NLU model without your changes to an NLU model with your
changes.

You can do this by running NLU testing in cross-validation mode:

.. code-block:: bash

   rasa test nlu --cross-validation

You could also train a model on a training set and testing it on a test set. If you use the train-test
set approach, it is best to :ref:`shuffle and split your data <train-test-split>` using ``rasa data split`` as part of this CI step, as
opposed to using a static NLU test set, which can easily become outdated.

Because this test doesn't result in a pass/fail exit code, it's best to make
the results visible so that you can interpret them.
For example, `this workflow <https://gist.github.com/amn41/de555c93913a01fbd56df2e2d211862c>`_
includes commenting on a PR with a results table that shows which intents are confused with others.

Since NLU comparison can be a fairly resource intensive test, you may choose to run this test
only when certain conditions are met. Conditions might include the presence of a manual label (e.g. "NLU
testing required"), changes to NLU data, or changes to the NLU pipeline.

.. _testing-action-code:

Test Action Code
################

The approach used to test your action code will depend on how it is
implemented. For example, if you connect to external APIs, it is recommended to write unit tests to ensure
that those APIs respond as expected to common inputs. However you test your action code, you should
include these tests in your CI pipeline so that they run each time you make changes.

Continuous Deployment (CD)
--------------------------

To get improvements out to your users frequently, you will want to automate as
much of the deployment process as possible.

CD steps usually run on push or merge to a certain branch, once CI checks have
succeeded.

.. contents::
   :local:

.. _uploading-a-model:

Deploy your Rasa Model
######################

If you ran :ref:`end-to-end tests <test-the-assistant>` in your CI pipeline,
you'll already have a trained model. You can set up your CD pipeline to upload the trained model to your
Rasa server if the CI results are satisfactory. For example, to upload a model to Rasa X:

.. code-block:: bash

   curl -k -F "model=@models/my_model.tar.gz" "https://example.rasa.com/api/projects/default/models?api_token={your_api_token}"

If you are using Rasa X, you can also `tag the uploaded model <https://rasa.com/docs/rasa-x/api/rasa-x-http-api/#tag/Models/paths/~1projects~1{project_id}~1models~1{model}~1tags~1{tag}/put>`_
as ``active`` (or whichever deployment you want to tag if using multiple `deployment environments <https://rasa.com/docs/rasa-x/enterprise/deployment-environments/#>`_):

.. code-block:: bash

   curl -X PUT "https://example.rasa.com/api/projects/default/models/my_model/tags/active"


However, if your update includes changes to both your model and your action
code, and these changes depend on each other in any way, you should **not**
automatically tag the model as ``production``. You will first need to build and
deploy your updated action server, so that the new model won't e.g. call
actions that don't exist in the pre-update action server.

Deploy your Action Server
#########################

You can automate
:ref:`building and uploading a new image for your action server <building-an-action-server-image>`,
to an image repository for each
update to your action code. As noted above, you should be careful with
automatically deploying a new image tag to production if the action server
would be incompatible with the current production model.

Example CI/CD pipelines
-----------------------

As examples, see the CI/CD pipelines for
`Sara <https://github.com/RasaHQ/rasa-demo/blob/master/.github/workflows/build_and_deploy.yml>`_,
the Rasa assistant that you can talk to in the Rasa Docs, and
`Carbon Bot <https://github.com/RasaHQ/carbon-bot/blob/master/.github/workflows/model_ci.yml>`_.
Both use `Github Actions <https://github.com/features/actions>`_ as a CI/CD tool.

These examples are just two of many possibilities. If you have a CI/CD setup you like, please
share it with the Rasa community on the `forum <https://forum.rasa.com>`_.
