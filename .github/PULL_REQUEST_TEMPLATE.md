**Proposed changes**:
- ...

**Status (please check what you already did)**:
- [ ] added some tests for the functionality
- [ ] updated the documentation
- [ ] updated the changelog (please check [changelog](https://github.com/RasaHQ/rasa-private/tree/main/changelog) for instructions)
- [ ] reformat files using `ruff` (please check [Readme](https://github.com/RasaHQ/rasa-private#code-style) for instructions)

**Dependency or Vulnerability Upgrade Checklist**:
- [ ] ran the [installation time measurement workflow](https://github.com/RasaHQ/rasa-private/blob/main/.github/workflows/run-performance-checks-on-main.yml) before every dependency upgrade (you can use empty commits to trigger it or you can check the [Metabase query](https://rasa.metabaseapp.com/question/1245-rasa-pro-installation-time-with-pip) for historical context)
- [ ] ran the above-mentioned workflow after every dependency upgrade (you can trigger it manually by navigating to the [GH workflow](https://github.com/RasaHQ/rasa-private/actions/workflows/run-performance-checks-on-main.yml) and clicking "Run workflow" on your remote branch)
- [ ] if you are planning to make a rasa-sdk release (dev, rc, micro etc.) to support changes in rasa-pro, please make sure to upgrade dependencies in rasa-sdk first before releasing rasa-sdk.

**LLM and Embeddings providers' E2E tests checklist**

- [ ] If your changes effect integration/compatibility with providers, then please run the [Providers E2E tests workflow](./workflows/providers-e2e-tests.yml) on your branch.
- [ ] If required, update the [provider(s)' test configs](../data/test_config/providers/) as appropriate.
