## 📂 `data/` – The Agent's Business Logic

This folder holds files that define your agent’s skills using "flows". [1](https://rasa.com/docs/pro/build/writing-flows) Each flow is a step-by-step conversation pattern (like a recipe) for handling a user goal (e.g., checking a balance, updating an address).

**What you'll find:**
- **accounts/**: Account management flows (balance checking, statement downloads)
- **bills/**: Bill payment flows and reminders
- **cards/**: Card management flows (activation, blocking, replacement, listing)
- **contacts/**: Contact management flows (add, list, remove trusted contacts)
- **transfers/**: Money transfer flows (account-to-account, third-party payments)
- **general/**: General banking conversations (greetings, help, support)

**Edit YAML files in this folder** to add new banking features, modify existing flows, or adjust what the agent asks customers.

