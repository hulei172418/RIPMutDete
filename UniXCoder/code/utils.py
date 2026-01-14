import logging
from typing import Dict, List, Tuple
import os
import json

from tree_sitter_languages import get_parser

from GraphCodeBERT.code.parser.DFG import (
    DFG_python,
    DFG_java,
    DFG_ruby,
    DFG_go,
    DFG_php,
    DFG_javascript,
)
from GraphCodeBERT.code.parser import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
)

logger = logging.getLogger(__name__)

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
}

parsers: Dict[str, Tuple] = {}
for lang, dfg_fn in dfg_function.items():
    ts_parser = get_parser(lang)
    parsers[lang] = (ts_parser, dfg_fn)


def extract_dataflow(code: str, lang: str = "java") -> Tuple[List[str], List[Tuple]]:
    """
    Remove comments, tokenize the code, and extract data-flow information.
    Returns (code_tokens, dfg).
    """
    parser, dfg_fn = parsers[lang]

    # 1) Strip comments/docstrings
    try:
        code = remove_comments_and_docstrings(code, lang)
    except Exception:
        pass

    # 2) PHP code needs to be wrapped with tags
    if lang == "php":
        code = "<?php" + code + "?>"

    # 3) Build AST & DFG
    try:
        tree = parser.parse(code.encode("utf8"))
        root_node = tree.root_node

        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split("\n")
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]

        index_to_code = {}
        for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, tok)

        try:
            dfg, _ = dfg_fn(root_node, index_to_code, {})
        except Exception:
            dfg = []

        dfg = sorted(dfg, key=lambda x: x[1])
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_dfg = []
        for d in dfg:
            if d[1] in indexs:
                new_dfg.append(d)
        dfg = new_dfg
    except Exception:
        code_tokens, dfg = [], []

    return code_tokens, dfg


def get_max_position_embeddings(config_file_path: str, default_value: int = 4096) -> int:
    if not os.path.exists(config_file_path):
        return default_value
    with open(config_file_path, "r", encoding="utf8") as f:
        config = json.load(f)
    return config.get("max_position_embeddings", default_value)