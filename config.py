MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 50
SEQUENCE_LENGTHS = [64, 128, 256]

FP32_TOKENS = 10    # scheduling threshold

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

MODES = ["fp32", "int8", "scheduled"]