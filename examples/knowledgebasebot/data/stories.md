## Happy path 1
* greet
  - utter_greet
* query_knowledge_base
  - action_query_knowledge_base
* goodbye
  - utter_goodbye

## Happy path 2
* greet
  - utter_greet
* query_knowledge_base
  - action_query_knowledge_base
* query_knowledge_base
  - action_query_knowledge_base
* goodbye
  - utter_goodbye

## Hello
* greet
- utter_greet

## Query Knowlege Base
* query_knowledge_base
- action_query_knowledge_base
- slot{"attribute": null}
- slot{"object_type": null}
- slot{"mention": null}

## Bye
* goodbye
- utter_goodbye
