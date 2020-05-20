# constants for configuration parameters of our tensorflow models

LABEL = "label"
HIDDEN_LAYERS_SIZES = "hidden_layers_sizes"
SHARE_HIDDEN_LAYERS = "share_hidden_layers"

TRANSFORMER_SIZE = "transformer_size"
NUM_TRANSFORMER_LAYERS = "number_of_transformer_layers"
NUM_HEADS = "number_of_attention_heads"
UNIDIRECTIONAL_ENCODER = "unidirectional_encoder"
KEY_RELATIVE_ATTENTION = "use_key_relative_attention"
VALUE_RELATIVE_ATTENTION = "use_value_relative_attention"
MAX_RELATIVE_POSITION = "max_relative_position"

BATCH_SIZES = "batch_size"
BATCH_STRATEGY = "batch_strategy"
EPOCHS = "epochs"
RANDOM_SEED = "random_seed"
LEARNING_RATE = "learning_rate"

DENSE_DIMENSION = "dense_dimension"
EMBEDDING_DIMENSION = "embedding_dimension"

SIMILARITY_TYPE = "similarity_type"
LOSS_TYPE = "loss_type"
NUM_NEG = "number_of_negative_examples"
MAX_POS_SIM = "maximum_positive_similarity"
MAX_NEG_SIM = "maximum_negative_similarity"
USE_MAX_NEG_SIM = "use_maximum_negative_similarity"

SCALE_LOSS = "scale_loss"
REGULARIZATION_CONSTANT = "regularization_constant"
NEGATIVE_MARGIN_SCALE = "negative_margin_scale"
DROP_RATE = "drop_rate"
DROP_RATE_ATTENTION = "drop_rate_attention"
DROP_RATE_DIALOGUE = "drop_rate_dialogue"
DROP_RATE_LABEL = "drop_rate_label"

WEIGHT_SPARSITY = "weight_sparsity"

EVAL_NUM_EPOCHS = "evaluate_every_number_of_epochs"
EVAL_NUM_EXAMPLES = "evaluate_on_number_of_examples"

INTENT_CLASSIFICATION = "intent_classification"
ENTITY_RECOGNITION = "entity_recognition"
MASKED_LM = "use_masked_language_model"

SPARSE_INPUT_DROPOUT = "use_sparse_input_dropout"
DENSE_INPUT_DROPOUT = "use_dense_input_dropout"

RANKING_LENGTH = "ranking_length"

BILOU_FLAG = "BILOU_flag"

RETRIEVAL_INTENT = "retrieval_intent"

SOFTMAX = "softmax"
MARGIN = "margin"
AUTO = "auto"
INNER = "inner"
COSINE = "cosine"

BALANCED = "balanced"
SEQUENCE = "sequence"

POOLING = "pooling"
MAX_POOLING = "max"
MEAN_POOLING = "mean"

TENSORBOARD_LOG_DIR = "tensorboard_log_directory"
TENSORBOARD_LOG_LEVEL = "tensorboard_log_level"
