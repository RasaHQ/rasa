:desc: Set up a CI/CD pipeline to ensure that iterative improvements to your assistant are tested a
       and deployed with minimum manual effort a
 a
.. _setting-up-ci-cd: a
 a
Setting up CI/CD a
================ a
 a
Even though developing a contextual assistant is different from developing traditional a
software, you should still follow software development best practices. a
Setting up a Continuous Integration (CI) and Continuous Deployment (CD) a
pipeline ensures that incremental updates to your bot are improving it, not harming it. a
 a
.. contents:: a
   :local: a
   :depth: 2 a
 a
 a
Overview a
-------- a
 a
Continous Integration (CI) is the practice of merging in code changes a
frequently and automatically testing changes as they are committed. Continuous a
Deployment (CD) means automatically deploying integrated changes to a staging a
or production environment. Together, they allow you to make more frequent improvements a
to your assistant and efficiently test and deploy those changes. a
 a
This guide will cover **what** should go in a CI/CD pipeline, specific to a a
Rasa project. **How** you implement that pipeline is up to you. a
There are many CI/CD tools out there, such as `GitHub Actions <https://github.com/features/actions>`_, a
`GitLab CI/CD <https://docs.gitlab.com/ee/ci/>`_, `Jenkins <https://www.jenkins.io/doc/>`_, and a
`CircleCI <https://circleci.com/docs/2.0/>`_. We recommend choosing a tool that integrates with a
whatever Git repository you use. a
 a
 a
Continuous Integration (CI) a
--------------------------- a
 a
The best way to improve an assistant is with frequent `incremental updates a
<https://rasa.com/docs/rasa-x/user-guide/improve-assistant/>`_. a
No matter how small a change is, you want to be sure that it doesn't introduce a
new problems or negatively impact the performance of your assistant. a
 a
It is usually best to run CI checks on merge / pull requests or on commit. Most tests are a
quick enough to run on every change. However, you can choose to run more a
resource-intensive tests only when certain files have been changed or when some a
other indicator is present. For example, if your code is hosted on Github, a
you can make a test run only if the pull request has a certain label (e.g. "NLU testing required"). a
 a
.. contents:: a
   :local: a
 a
Validate Data and Stories a
######################### a
 a
:ref:`Data validation <validate-files>` verifies that there are no mistakes or a
major inconsistencies in your domain file, NLU data, or story data. a
 a
.. code-block:: bash a
 a
   rasa data validate --fail-on-warnings --max-history <max_history> a
 a
If data validation results in errors, training a model will also fail. By a
including the ``--fail-on-warnings`` flag, validation will also fail on a
warnings about problems that won't prevent training a model, but might indicate a
messy data, such as actions listed in the domain that aren't used in any a
stories. a
 a
Data validation includes :ref:`story structure validation <test-story-files-for-conflicts>`. a
Story validation checks if you have any a
stories where different bot actions follow from the same dialogue history. a
Conflicts between stories will prevent a model from learning the correct a
pattern for a dialogue. Set the ``--max-history`` parameter to the value of ``max_history`` for the a
memoization policy in your ``config.yml``. If you haven't set one, use the default of ``5``. a
 a
Train a Model a
############# a
 a
.. code-block:: bash a
 a
   rasa train a
 a
Training a model verifies that your NLU pipeline and policy configurations are a
valid and trainable, and it provides a model to use for test conversations. a
If it passes the CI tests, then you can also :ref:`upload the trained model <uploading-a-model>` a
to your server as part of the continuous deployment process . a
 a
.. _test-the-assistant: a
 a
Test the Assistant a
################## a
 a
Testing your trained model on :ref:`test conversations a
<end-to-end-testing>` is the best way to have confidence in how your assistant a
will act in certain situations. These stories, written in a modified story a
format, allow you to provide entire conversations and test that, given this a
user input, your model will behave in the expected manner. This is especially a
important as you start introducing more complicated stories from user a
conversations. a
 a
 a
.. code-block:: bash a
 a
   rasa test --stories tests/conversation_tests.md --fail-on-prediction-errors a
 a
The ``--fail-on-prediction-errors`` flag ensures the test will fail if any test a
conversation fails. a
 a
End-to-end testing is only as thorough and accurate as the test a
cases you include, so you should continue to grow your set of test conversations a
as you make improvements to your assistant. A good rule of thumb to follow is that you should aim for your test conversations a
to be representative of the true distribution of real conversations. a
Rasa X makes it easy to `add test conversations based on real conversations <https://rasa.com/docs/rasa-x/user-guide/improve-assistant.html#add-test-conversation>`_. a
 a
Note: End-to-end testing does **not** execute your action code. You will need to a
:ref:`test your action code <testing-action-code>` in a seperate step. a
 a
Compare NLU Performance a
####################### a
 a
If you've made significant changes to your NLU training data (e.g. a
splitting an intent into two intents or adding a lot of training examples), you should run a a
:ref:`full NLU evaluation <nlu-evaluation>`. You'll want to compare a
the performance of the NLU model without your changes to an NLU model with your a
changes. a
 a
You can do this by running NLU testing in cross-validation mode: a
 a
.. code-block:: bash a
 a
   rasa test nlu --cross-validation a
 a
You could also train a model on a training set and testing it on a test set. If you use the train-test a
set approach, it is best to :ref:`shuffle and split your data <train-test-split>` using ``rasa data split`` as part of this CI step, as a
opposed to using a static NLU test set, which can easily become outdated. a
 a
Because this test doesn't result in a pass/fail exit code, it's best to make a
the results visible so that you can interpret them. a
For example, `this workflow <https://gist.github.com/amn41/de555c93913a01fbd56df2e2d211862c>`_ a
includes commenting on a PR with a results table that shows which intents are confused with others. a
 a
Since NLU comparison can be a fairly resource intensive test, you may choose to run this test a
only when certain conditions are met. Conditions might include the presence of a manual label (e.g. "NLU a
testing required"), changes to NLU data, or changes to the NLU pipeline. a
 a
.. _testing-action-code: a
 a
Test Action Code a
################ a
 a
The approach used to test your action code will depend on how it is a
implemented. For example, if you connect to external APIs, it is recommended to write unit tests to ensure a
that those APIs respond as expected to common inputs. However you test your action code, you should a
include these tests in your CI pipeline so that they run each time you make changes. a
 a
Continuous Deployment (CD) a
-------------------------- a
 a
To get improvements out to your users frequently, you will want to automate as a
much of the deployment process as possible. a
 a
CD steps usually run on push or merge to a certain branch, once CI checks have a
succeeded. a
 a
.. contents:: a
   :local: a
 a
.. _uploading-a-model: a
 a
Deploy your Rasa Model a
###################### a
 a
If you ran :ref:`end-to-end tests <test-the-assistant>` in your CI pipeline, a
you'll already have a trained model. You can set up your CD pipeline to upload the trained model to your a
Rasa server if the CI results are satisfactory. For example, to upload a model to Rasa X: a
 a
.. code-block:: bash a
 a
   curl -k -F "model=@models/my_model.tar.gz" "https://example.rasa.com/api/projects/default/models?api_token={your_api_token}" a
 a
If you are using Rasa X, you can also `tag the uploaded model <https://rasa.com/docs/rasa-x/api/rasa-x-http-api/#tag/Models/paths/~1projects~1{project_id}~1models~1{model}~1tags~1{tag}/put>`_ a
as ``active`` (or whichever deployment you want to tag if using multiple `deployment environments <https://rasa.com/docs/rasa-x/enterprise/deployment-environments/#>`_): a
 a
.. code-block:: bash a
 a
   curl -X PUT "https://example.rasa.com/api/projects/default/models/my_model/tags/active" a
 a
 a
However, if your update includes changes to both your model and your action a
code, and these changes depend on each other in any way, you should **not** a
automatically tag the model as ``production``. You will first need to build and a
deploy your updated action server, so that the new model won't e.g. call a
actions that don't exist in the pre-update action server. a
 a
Deploy your Action Server a
######################### a
 a
You can automate a
:ref:`building and uploading a new image for your action server <building-an-action-server-image>`, a
to an image repository for each a
update to your action code. As noted above, you should be careful with a
automatically deploying a new image tag to production if the action server a
would be incompatible with the current production model. a
 a
Example CI/CD pipelines a
----------------------- a
 a
As examples, see the CI/CD pipelines for a
`Sara <https://github.com/RasaHQ/rasa-demo/blob/master/.github/workflows/build_and_deploy.yml>`_, a
the Rasa assistant that you can talk to in the Rasa Docs, and a
`Carbon Bot <https://github.com/RasaHQ/carbon-bot/blob/master/.github/workflows/model_ci.yml>`_. a
Both use `Github Actions <https://github.com/features/actions>`_ as a CI/CD tool. a
 a
These examples are just two of many possibilities. If you have a CI/CD setup you like, please a
share it with the Rasa community on the `forum <https://forum.rasa.com>`_. a
 a