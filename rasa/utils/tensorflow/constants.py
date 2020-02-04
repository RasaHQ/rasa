# constants - configuration parameters

HIDDEN_LAYERS_SIZES_TEXT = "hidden_layers_sizes_text"
HIDDEN_LAYERS_SIZES_LABEL = "hidden_layers_sizes_label"
HIDDEN_LAYERS_SIZES_DIALOGUE = "hidden_layers_sizes_dialogue"
SHARE_HIDDEN_LAYERS = "share_hidden_layers"

TRANSFORMER_SIZE = "transformer_size"
NUM_TRANSFORMER_LAYERS = "number_of_transformer_layers"
NUM_HEADS = "number_of_attention_heads"
UNIDIRECTIONAL_ENCODER = "unidirectional_encoder"

POS_ENCODING = "positional_encoding"
MAX_SEQ_LENGTH = "maximum_sequence_length"

BATCH_SIZES = "batch_sizes"
BATCH_STRATEGY = "batch_strategy"
EPOCHS = "epochs"
RANDOM_SEED = "random_seed"
LEARNING_RATE = "learning_rate"

DENSE_DIM = "dense_dimensions"
EMBED_DIM = "embedding_dimension"

SIMILARITY_TYPE = "similarity_type"
LOSS_TYPE = "loss_type"
NUM_NEG = "number_of_negative_examples"
MU_POS = "maximum_positive_similarity"
MU_NEG = "maximum_negative_similarity"
USE_MAX_SIM_NEG = "use_maximum_negative_similarity"

SCALE_LOSS = "scale_loss"
REGULARIZATION_CONSTANT = "regularization_constant"
NEG_MARGIN_SCALE = "neg_margin_scale"
DROPRATE = "droprate"
DROPRATE_ATTENTION = "droprate_attention"
DROPRATE_DIALOGUE = "droprate_dialogue"
DROPRATE_LABEL = "droprate_label"

EVAL_NUM_EPOCHS = "evaluate_every_number_of_epochs"
EVAL_NUM_EXAMPLES = "evaluate_on_number_of_examples"

INTENT_CLASSIFICATION = "perform_intent_classification"
ENTITY_RECOGNITION = "perform_entity_recognition"
MASKED_LM = "use_masked_language_model"

SPARSE_INPUT_DROPOUT = "use_sparse_input_dropout"

RANKING_LENGTH = "ranking_length"

BILOU_FLAG = "BILOU_flag"

KEY_RELATIVE_ATTENTION = "use_key_relative_attention"
VALUE_RELATIVE_ATTENTION = "use_value_relative_attention"
MAX_RELATIVE_POSITION = "max_relative_position"
