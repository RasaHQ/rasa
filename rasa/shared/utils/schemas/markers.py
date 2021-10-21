# MARKER_ID_SCHEMA = {
#     "description": "The unique identifier for a marker.",
#     "type": "string",
# }

# OPERATOR_SCHEMA = {
#     "description": "The operator to apply to the list of conditions.",
#     "type": "string",
#     "enum": ["AND", "OR", "SEQ"],
#     "default": "AND",
# }

# EVENT_SCHEMA = {
#     "description": "A list of events nested under a condition.",
#     "type": "array",
#     "items": {"type": "string"},
#     "minItems": 1,
#     "uniqueItems": True,
# }


# # FIXME: negation operator should be somewhere here to avoid set and "not set"

# CONDITION_SCHEMA = {
#     "description": "The list of conditions for a marker",
#     "type": "array",
#     "items": {
#         "type": "object",
#         "properties": {
#             "set": EVENT_SCHEMA,
#             "type": # FIXME: action, intent, ... here
#         },
#         "additionalProperties": False,
#     },
#     "minItems": 1,
# }


# MARKERS_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "markers": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "marker": MARKER_ID_SCHEMA,
#                     "operator": OPERATOR_SCHEMA,
#                     "condition": # list of marker schemas or CONDITION_SCHEMA
#                 },
#                 "required": ["marker", "condition"],
#             },
#         },
#     },
#     "required": ["markers"],
# }
