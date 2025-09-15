## 📂 `domain/` – The Agent's Brain

This folder contains YAML files that define the agent's:
- **Slots**: The agent's memory (e.g., user's name, session info) [1](https://rasa.com/docs/reference/primitives/slots/).
- **Responses**: The messages your agent can say to users (e.g., greetings, confirmations) [2](https://rasa.com/docs/reference/primitives/responses).
- **Actions**: Custom logic your agent can run (e.g., handling feedback, human handoff) [3](https://rasa.com/docs/reference/primitives/custom-actions).

**What you'll find:**
- **general/**: Basic conversational domain elements (greetings, help, feedback)

You can organize the domain as one big file or many small ones. Rasa will automatically merge everything during training [1](https://rasa.com/docs/reference/config/domain).
