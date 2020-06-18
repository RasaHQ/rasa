# Rasa NLU: Language Understanding for Chatbots and AI assistants

Rasa NLU is an open-source natural language processing tool for intent classification, response retrieval and
entity extraction in chatbots. For example, taking a sentence like

```
"I am looking for a Mexican restaurant in the center of town"
```

and returning structured data like

```
{
  "intent": "search_restaurant",
  "entities": {
    "cuisine" : "Mexican",
    "location" : "center"
  }
}
```

If you want to use Rasa NLU on its own, see Using NLU Only.
