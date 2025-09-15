## 📂 `data/` – The Agent's Business Logic

This folder holds files that define your agent’s skills using "flows". [1](https://rasa.com/docs/pro/build/writing-flows) Each flow is a step-by-step conversation pattern (like a recipe) for handling a user goal (e.g., checking a balance, updating an address).

**What you'll find:**
- **billing/**: Billing and payment flows (bill inquiries, payment processing)
- **network/**: Network service flows (troubleshooting, diagnostics, service checks)
- **general/**: General telecom conversations (greetings, help, support)

**Edit YAML files in this folder** to add new telecom features, modify existing flows, or adjust what the agent asks customers.

