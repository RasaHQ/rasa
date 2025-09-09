## 📂 `actions/` – The Agent's Toolkit

This folder contains Python files that define custom banking actions your agent can perform. Actions handle complex banking operations, database interactions, and business logic validation.

**What you'll find:**
- **accounts/**: Account-related actions (balance checking, account validation)
- **cards/**: Card management actions (card existence checks, status updates)
- **transfers/**: Payment and transfer actions (payee management, fund transfers, scheduling)
- **contacts/**: Contact and advisor management
- `database.py`: Core database interactions with banking data
- `action_session_start.py`: Session initialization for banking context

**Edit Python files in this folder** to add new banking features, modify business rules, or integrate with external banking APIs and systems.

Learn more about custom actions in the [Rasa documentation](https://rasa.com/docs/pro/build/custom-actions).
