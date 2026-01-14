import os
import json

import torch
import random
import numpy as np
from transformers import AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_max_position_embeddings(config_file_path: str, default_value: int = 4096) -> int:
    if not os.path.exists(config_file_path):
        return default_value
    with open(config_file_path, "r", encoding="utf8") as f:
        config = json.load(f)
    return config.get("max_position_embeddings", default_value)


def truncate_code_to_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)
