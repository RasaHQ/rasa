## 📂 `domain/` – The Agent's Brain

This folder contains YAML files that define the telecom agent's:
- **Slots**: The agent's memory (e.g., customer account, service status, network diagnostics) [1](https://rasa.com/docs/reference/primitives/slots/).
- **Responses**: The messages your agent can say to customers (e.g., greetings, service confirmations, technical explanations) [2](https://rasa.com/docs/reference/primitives/responses).
- **Actions**: Custom logic your agent can run (e.g., running network diagnostics, checking service status) [3](https://rasa.com/docs/reference/primitives/custom-actions).

**What you'll find:**
- **billing/**: Domain configuration for billing and payment features
- **network/**: Domain setup for network diagnostics and troubleshooting
- **general/**: General telecom customer service domain elements

You can organize the domain as one big file or many small ones. Rasa will automatically merge everything during training [1](https://rasa.com/docs/reference/config/domain).
