<h1 align="center">Rasa Open Source</h1>

<div align="center">

[![Join the Agent Engineering Community](https://img.shields.io/badge/Community-Join%20the%20Discussion-blueviolet)](https://info.rasa.com/community?utm_source=github&utm_medium=website&utm_campaign=)
[![Try Hello Rasa](https://img.shields.io/badge/Playground-Try%20Hello%20Rasa-ff69b4)](https://hello.rasa.ai/?utm_source=github&utm_medium=website&utm_campaign=)
[![PyPI version](https://badge.fury.io/py/rasa.svg)](https://badge.fury.io/py/rasa)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/rasa.svg)](https://pypi.python.org/pypi/rasa)
[![Build Status](https://github.com/RasaHQ/rasa/workflows/Continuous%20Integration/badge.svg)](https://github.com/RasaHQ/rasa/actions)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs)

</div>

<br />

<div align="center">
  <h3>🚧 <b>Note: Maintenance Mode</b> 🚧</h3>
  <p>
    Rasa Open Source is currently in maintenance mode. 
    <br />
    The future of building AI agents with Rasa is <b>Hello Rasa</b> and <b>CALM</b>.
  </p>
</div>

<hr />

## 🚀 The Future of Rasa: Hello Rasa

**Building reliable AI agents just got easier.**

[**Hello Rasa**](https://hello.rasa.ai/?utm_source=github&utm_medium=website&utm_campaign=) is our new interactive playground for prototyping AI agents. It combines LLM fluency with the reliability of business logic using our **CALM** (Conversational AI with Language Models) engine.

### Why switch to Hello Rasa?

* **No setup required:** Open the playground, pick a template (Banking, Telecom, Support), and start building in your browser.
* **No NLU training:** We have moved beyond intents. The LLM handles dialogue understanding while you define the business flows.
* **Built-in copilot:** A specialized AI assistant helps you generate code, debug flows, and expand your agent instantly.
* **Production ready:** Hello Rasa is not just a toy. Export your agent to the Rasa Platform when you are ready to scale.

### Core concepts

* **CALM:** Combines LLM flexibility with strict business logic. The LLM understands the user; the code enforces the rules.
* **Flows:** Describe logical steps (e.g., collect money, transfer funds) rather than rigid dialogue trees.
* **Inspector:** See real-time decision-making. No black boxes.

👉 **[Start building for free at Hello Rasa](https://hello.rasa.ai/?utm_source=github&utm_medium=website&utm_campaign=)**

---

## 🧠 Join the Agent Engineering Community

We are building a home for people shipping real-world AI agents. 

Agent Engineering is evolving faster than any single framework. This is a vendor-neutral space to discuss architectures, memory, orchestration, and safety with builders across the industry.

### What you get:
* **Network:** Meet engineers building production agents
* **Learn:** Discuss practical patterns, not just theory
* **Access:** Direct influence on the Hello Rasa roadmap and early access to features

| Channel | Purpose |
| :--- | :--- |
| **#agent-design** | Architectures, reasoning, memory, testing |
| **#showcase** | Show your builds, demos, and repos |
| **#ask-anything** | Debugging and workflow questions |

👉 **[Join the Community](https://info.rasa.com/community?utm_source=github&utm_medium=website&utm_campaign=)**

---

<br>
<br>

# Rasa Open Source (Legacy)

> **Note:** The documentation and installation instructions below apply to the classic Rasa Open Source framework. For the latest CALM-based experience, see the [Hello Rasa](#-the-future-of-rasa-hello-rasa) section above.

Rasa is an open source machine learning framework for automating text and voice-based conversations. With Rasa, you can build contextual assistants on:

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

Rasa helps you build contextual assistants that can handle layered conversations with lots of back-and-forth. 

### 📚 Resources
- 🤓 [Read the docs](https://rasa.com/docs/rasa/)
- 😁 [Install Rasa](https://rasa.com/docs/rasa/installation/environment-set-up)
- 🚀 [Learn all about Conversational AI](https://learning.rasa.com/)
- 🏢 [Explore the enterprise platform](https://rasa.com/product/rasa-platform/)

## Development Internals & Contributing

We are happy to receive contributions. Please review our [Contribution Guidelines](CONTRIBUTING.md) before getting started.

### Installation for Development
Rasa uses **Poetry** for packaging and dependency management.

1.  **Install Poetry**: Follow the [official guide](https://python-poetry.org/docs/#installation).
2.  **Build from source**:
    ```bash
    make install
    ```
    *Note for macOS users*: If you run into compiler issues, try `export SYSTEM_VERSION_COMPAT=1` before installation.

### Running Tests
Make sure you have development requirements installed:

```bash
make prepare-tests-ubuntu # Ubuntu/Debian
make prepare-tests-macos  # macOS
make test                 # Run tests
```

### Releases

Rasa follows Semantic Versioning.

  * **Major**: Incompatible API changes
  * **Minor**: Backward-compatible functionality
  * **Patch**: Backward-compatible bug fixes

For full details on our release cadence and maintenance policy, visit our [Product Release and Maintenance Policy](https://rasa.com/rasa-product-release-and-maintenance-policy/).

## License

Licensed under the Apache License, Version 2.0. Copyright 2022 Rasa Technologies GmbH. [Copy of the license](https://www.google.com/search?q=LICENSE.txt).
