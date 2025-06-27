**Proposed changes**:
- ...

**Status (please check what you already did)**:
- [ ] added some tests for the functionality
- [ ] updated the documentation
- [ ] updated the changelog (please check [changelog](https://github.com/RasaHQ/rasa-private/tree/main/changelog) for instructions)
- [ ] reformat files using `ruff` (please check [Readme](https://github.com/RasaHQ/rasa-private/blob/main/README-INTERNAL.md#code-style) for instructions)

**Dependency or Vulnerability Upgrade Checklist**:
- [ ] before every upgrade, run the [installation time measurement workflow](https://github.com/RasaHQ/rasa-private/blob/main/.github/workflows/run-performance-checks-on-main.yml) (you can use empty commits to trigger it, or you can check the [Metabase query](https://rasa.metabaseapp.com/question/1245-rasa-pro-installation-time-with-pip) for historical context)
- [ ] follow the best practices detailed in the `Dependency Management` section of the [internal README](https://github.com/RasaHQ/rasa-private/blob/main/README-INTERNAL.md)
- [ ] compare this measurement against the one in the automated PR comment after the automated workflow runs
- [ ] if you are planning to make a rasa-sdk release (dev, rc, micro etc.) to support changes in rasa-pro, please make sure to upgrade dependencies in rasa-sdk first before releasing rasa-sdk.

**LLM and Embeddings providers' E2E tests checklist**

- [ ] If your changes effect integration/compatibility with providers, then please run the [Providers E2E tests workflow](./workflows/providers-e2e-tests.yml) on your branch.
- [ ] If required, update the [provider(s)' test configs](../data/test_config/providers/) as appropriate.

**PII Leakage Detection and Prevention Checklist**
- [ ] **String formatting** - Confirm that string interpolation (f-strings, .format(), %) do not include variables that may contain PII.
- [ ] **Keyword arguments** - Check that logging calls do not pass variables that may contain PII as named parameters.
- [ ] **Debug vs other log levels** - Verify appropriate log levels are used if variables that may contain PII have to be logged (**Only debug logs are allowed to contain PII**).
- [ ] **Common PII sources** - Slot values, user messages, bot responses, tracker state, dialogue stack/frames, commands, and events.
- [ ] **Variable naming patterns** - Variables containing these keywords may contain PII: `slot`, `user`, `text`, `message`, `response`, `tracker`, `stack`, `frame`, `top`, `command`, `event`, `document`, `context`, `input`.
