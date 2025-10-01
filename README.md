<h1 align="center">Rasa</h1>

<div align="center">

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=RasaHQ_rasa&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=RasaHQ_rasa)
[![Documentation Status](https://img.shields.io/badge/docs-stable-brightgreen.svg)](https://rasa.com/docs/docs/pro/intro)
![Python version support](https://img.shields.io/pypi/pyversions/rasa-pro)

</div>

<hr />

Rasa is a framework for building scalable, dynamic conversational AI assistants that integrate large language models (LLMs) to enable more contextually aware and agentic interactions. Whether you’re new to conversational AI or an experienced developer, Rasa offers enhanced flexibility, control, and performance for mission-critical applications.

**Key Features:**

- **Flows for Business Logic:** Easily define business logic through Flows, a simplified way to describe how your AI assistant should handle conversations. Flows help streamline the development process, focusing on key tasks and reducing the complexity involved in managing conversations.
- **Automatic Conversation Repair:** Ensure seamless interactions by automatically handling interruptions or unexpected inputs. Developers have full control to customize these repairs based on specific use cases.
- **Customizable and Open:** Fully customizable code that allows developers to modify Rasa to meet specific requirements, ensuring flexibility and adaptability to various conversational AI needs.
- **Robustness and Control:** Maintain strict adherence to business logic, preventing unwanted behaviors like prompt injection and hallucinations, leading to more reliable responses and secure interactions.
- **Built-in Security:** Safeguard sensitive data, control access, and ensure secure deployment, essential for production environments that demand high levels of security and compliance. Secrets are managed through Pulumi's built-in secrets management system and can be integrated with HashiCorp Vault for enterprise-grade secret management.

A [free developer license](https://rasa.com/docs/pro/intro/#who-rasa-pro-is-for) is available so you can explore and get to know Rasa. It allows you to take your assistant live in production a limited capacity. A paid license is required for larger-scale production use, but all code is visible and can be customized as needed.

To get started right now, you can

`pip install rasa-pro`

Check out our

- [Rasa Quickstart](https://rasa.com/docs/learn/quickstart/pro),
- [Conversational AI with Language Models (CALM) conceptual rundown](https://rasa.com/docs/learn/concepts/calm),
- [Rasa tutorial](https://rasa.com/docs/pro/tutorial), and
- [Changelog](https://rasa.com/docs/reference/changelogs/rasa-pro-changelog)

for more. Also feel free to reach out to us on the [Rasa forum](https://forum.rasa.com/).

## Secrets Management

This project uses a multi-layered approach to secrets management:

- **Pulumi Secrets**: Primary secrets management through Pulumi's built-in configuration system (`pulumi.Config()`)
- **Kubernetes Secrets**: Application secrets are stored as Kubernetes secrets in the cluster
- **Vault Integration**: Optional HashiCorp Vault support for enterprise-grade secret management
- **AWS Secrets Manager**: Used selectively for specific services (e.g., database credentials in integration tests)

For infrastructure deployment, secrets are managed through Pulumi configuration files and environment variables, providing secure and flexible secret management across different deployment environments.
