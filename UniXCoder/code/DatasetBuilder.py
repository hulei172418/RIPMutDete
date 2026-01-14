import csv
import logging
import os
import pickle
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.serialization import add_safe_globals
from tqdm import tqdm

from UniXCoder.code.utils import extract_dataflow, parsers

import re
_RE_MUT = re.compile(r"\[MUT_CODE\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", re.S)
_RE_ORI = re.compile(r"\[ORIGIN_CODE\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", re.S)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """
    Feature representation of a single sample (two code snippets + DFG + label).
    Kept consistent with the original definition in run.py to be compatible
    with cached pickle/torch.save structures.
    """

    def __init__(
        self,
        input_tokens_1,
        input_ids_1,
        position_idx_1,
        dfg_to_code_1,
        dfg_to_dfg_1,
        input_tokens_2,
        input_ids_2,
        position_idx_2,
        dfg_to_code_2,
        dfg_to_dfg_2,
        label,
        url1,
        url2,
    ):
        # First code snippet
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1

        # Second code snippet
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2 = position_idx_2
        self.dfg_to_code_2 = dfg_to_code_2
        self.dfg_to_dfg_2 = dfg_to_dfg_2

        # Label and URLs
        self.label = label
        self.url1 = url1
        self.url2 = url2


def extract_code_from_rip(rip_text: str) -> str:
    if not rip_text:
        return ""
    m = _RE_MUT.search(rip_text)
    if m:
        return m.group(1).strip()
    m = _RE_ORI.search(rip_text)
    if m:
        return m.group(1).strip()
    return rip_text.strip()


def convert_examples_to_features(
    item: Tuple[str, str, int, Any, Any, Dict[str, Any], Dict[str, str]]
) -> InputFeatures:
    """
    Convert a tuple (url1, url2, label, tokenizer, args, cache, url_to_code)
    into an InputFeatures object.
    """
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    lang = "java"

    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]

            code_tokens, dfg = extract_dataflow(func, lang=lang)

            # Tokenization: prepend a virtual token "@ " for each line
            code_tokens = [
                tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
                for idx, x in enumerate(code_tokens)
            ]

            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            max_len = args.code_length + args.data_flow_length
            code_tokens = code_tokens[
                : max_len - 3 - min(len(dfg), args.data_flow_length)
            ][: 512 - 3]

            source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

            dfg = dfg[: max_len - len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for _ in dfg]
            source_ids += [tokenizer.unk_token_id for _ in dfg]

            padding_length = max_len - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length

            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            dfg = [
                x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
                for x in dfg
            ]
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

            cache[url] = (source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg)

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[url1]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[url2]

    return InputFeatures(
        source_tokens_1,
        source_ids_1,
        position_idx_1,
        dfg_to_code_1,
        dfg_to_dfg_1,
        source_tokens_2,
        source_ids_2,
        position_idx_2,
        dfg_to_code_2,
        dfg_to_dfg_2,
        label,
        url1,
        url2,
    )


add_safe_globals([InputFeatures])


class TextDataset(Dataset):
    """
    CSV-based equivalent mutant dataset wrapper.
    Responsibilities:
      - Read the url -> code mapping from code_db_file
      - Read (url1, url2, label) from train/eval/test CSV files
      - Call convert_examples_to_features to generate InputFeatures
      - Use torch.save / torch.load for feature caching
      - Build graph-guided attention masks in __getitem__
    """

    def __init__(self, tokenizer, args, file_path: str = "train"):
        postfix = file_path.split("/")[-1].split(".csv")[0]

        self.examples: List[InputFeatures] = []
        self.args = args
        index_filename = file_path

        logger.info("Creating features from index file at %s ", index_filename)

        url_to_code: Dict[str, str] = {}

        current_file_path = os.path.abspath(__file__)
        folder = os.path.join(os.path.dirname(current_file_path), "../cached_files")
        cache_file_path = os.path.join(folder, f"cached_{postfix}")
        code_pairs_file_path = os.path.join(folder, f"cached_{postfix}.pkl")
        code_pairs: List = []

        try:
            if not os.path.exists(folder):
                os.makedirs(folder)

            self.examples = torch.load(cache_file_path, map_location="cpu")
            with open(code_pairs_file_path, "rb") as f:
                code_pairs = pickle.load(f)
            logger.info("Loading features from cached file %s", cache_file_path)

            expected_len = self.args.code_length + self.args.data_flow_length
            try:
                example_len = len(self.examples[0].position_idx_1)
            except Exception:
                example_len = None
            if example_len != expected_len:
                raise ValueError(
                    f"Cached features use length {example_len}, "
                    f"but current args expect {expected_len}. Rebuild cache."
                )

        except Exception as e:
            logger.info(
                "Loading cached features failed (%s), creating features from dataset file at %s",
                repr(e),
                file_path,
            )

            with open(args.code_db_file, encoding="utf-8") as f:
                file_reader = csv.reader(f)
                _ = next(file_reader)
                for line in file_reader:
                    url_to_code[line[0]] = extract_code_from_rip(line[1])

            data: List[Tuple] = []
            cache: Dict[str, Any] = {}
            with open(index_filename, encoding="utf-8") as f:
                file_reader = csv.reader(f)
                _ = next(file_reader)
                for line in file_reader:
                    _, _, url1, url2, label = line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    label_int = 0 if label == "0" else 1
                    data.append((url1, url2, label_int, tokenizer, args, cache, url_to_code))

            for sing_example in data:
                code_pairs.append(
                    [
                        sing_example[0],
                        sing_example[1],
                        url_to_code[sing_example[0]],
                        url_to_code[sing_example[1]],
                    ]
                )
            with open(code_pairs_file_path, "wb") as f:
                pickle.dump(code_pairs, f)

            self.examples = [
                convert_examples_to_features(x) for x in tqdm(data, total=len(data))
            ]
            torch.save(self.examples, cache_file_path)

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: %s", idx)
                logger.info("label: %s", example.label)
                logger.info(
                    "input_tokens_1: %s",
                    [x.replace("\u0120", "_") for x in example.input_tokens_1],
                )
                logger.info("input_ids_1: %s", " ".join(map(str, example.input_ids_1)))
                logger.info("position_idx_1: %s", example.position_idx_1)
                logger.info("dfg_to_code_1: %s", " ".join(map(str, example.dfg_to_code_1)))
                logger.info("dfg_to_dfg_1: %s", " ".join(map(str, example.dfg_to_dfg_1)))

                logger.info(
                    "input_tokens_2: %s",
                    [x.replace("\u0120", "_") for x in example.input_tokens_2],
                )
                logger.info("input_ids_2: %s", " ".join(map(str, example.input_ids_2)))
                logger.info("position_idx_2: %s", example.position_idx_2)
                logger.info("dfg_to_code_2: %s", " ".join(map(str, example.dfg_to_code_2)))
                logger.info("dfg_to_dfg_2: %s", " ".join(map(str, example.dfg_to_dfg_2)))

    def __len__(self) -> int:
        return len(self.examples)

    def _build_attention_mask(
        self,
        position_idx: List[int],
        input_ids: List[int],
        dfg_to_code: List[Tuple[int, int]],
        dfg_to_dfg: List[List[int]],
    ) -> np.ndarray:
        """Build a graph-guided attention mask following the rules"""
        max_len = self.args.code_length + self.args.data_flow_length
        attn_mask = np.zeros((max_len, max_len), dtype=bool)

        node_index = sum(i > 1 for i in position_idx)
        max_length = sum(i != 1 for i in position_idx)

        attn_mask[:node_index, :node_index] = True

        for idx, token_id in enumerate(input_ids):
            if token_id in [0, 2]:
                attn_mask[idx, :max_length] = True

        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                if idx + node_index < max_len:
                    attn_mask[idx + node_index, a:b] = True
                    attn_mask[a:b, idx + node_index] = True

        for idx, nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(position_idx) and idx + node_index < max_len:
                    attn_mask[idx + node_index, a + node_index] = True

        return attn_mask

    def __getitem__(self, item: int):
        ex = self.examples[item]

        attn_mask_1 = self._build_attention_mask(
            ex.position_idx_1, ex.input_ids_1, ex.dfg_to_code_1, ex.dfg_to_dfg_1
        )
        attn_mask_2 = self._build_attention_mask(
            ex.position_idx_2, ex.input_ids_2, ex.dfg_to_code_2, ex.dfg_to_dfg_2
        )

        return (
            torch.tensor(ex.input_ids_1),
            torch.tensor(ex.position_idx_1),
            torch.tensor(attn_mask_1),
            torch.tensor(ex.input_ids_2),
            torch.tensor(ex.position_idx_2),
            torch.tensor(attn_mask_2),
            torch.tensor(ex.label),
        )
