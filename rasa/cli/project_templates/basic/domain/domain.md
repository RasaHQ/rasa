## 📂 `domain/` – The agent’s Brain

This folder contains YAML files that define the agent’s:
- **Slots**: The agent’s memory (e.g., user's name, account number, address).
- **Responses**: The messages your bot can say to users (e.g., greetings, confirmations).
- **Actions**: Custom logic your agent can run (e.g., checking a balance).


You can organize the domain as one big file or many small ones. Rasa will automatically merge everything during training [1](https://rasa.com/docs/reference/config/domain).
