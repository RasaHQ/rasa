## 📂 `domain/` – The Agent's Brain

This folder contains YAML files that define the agent's:
- **Slots**: The agent's memory (e.g., user's name, account number, address) [1](https://rasa.com/docs/reference/primitives/slots/).
- **Responses**: The messages your agent can say to users (e.g., greetings, confirmations) [2](https://rasa.com/docs/reference/primitives/responses).
- **Actions**: Custom logic your agent can run (e.g., checking a balance) [3](https://rasa.com/docs/reference/primitives/custom-actions).

The `bank_name` slot allows easy rebranding - just change its default value to customize all bank references throughout the agent.

**What you'll find:**
- **accounts/**: Domain configuration for account management features
- **bills/**: Domain setup for bill payment and reminder features
- **cards/**: Domain elements for card management operations
- **contacts/**: Domain configuration for contact and advisor management
- **transfers/**: Domain setup for money transfer and payment features
- **general/**: General banking conversation domain elements

You can organize the domain as one big file or many small ones. Rasa will automatically merge everything during training [1](https://rasa.com/docs/reference/config/domain).
