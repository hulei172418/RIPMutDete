import os
import pickle
import logging
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset

import re
_RE_MUT = re.compile(r"\[MUT_CODE\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", re.S)
_RE_ORI = re.compile(r"\[ORIGIN_CODE\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", re.S)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        label,
        url1,
        url2,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
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
    code1_tokens,
    code2_tokens,
    label,
    url1,
    url2,
    tokenizer,
    args,
    cache,
) -> InputFeatures:
    code1_tokens = code1_tokens[: args.code_length - 2]
    code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
    code2_tokens = code2_tokens[: args.code_length - 2]
    code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.code_length - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.code_length - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length

    source_tokens = code1_tokens + code2_tokens
    source_ids = code1_ids + code2_ids
    return InputFeatures(source_tokens, source_ids, label, url1, url2)


def get_example(item: Tuple[Any, ...]) -> InputFeatures:
    url1, url2, label, tokenizer, args, cache, url_to_code = item

    if url1 in cache:
        code1 = cache[url1].copy()
    else:
        try:
            code = " ".join(url_to_code[url1].split())
        except Exception:
            code = ""
        code1 = tokenizer.tokenize(code)

    if url2 in cache:
        code2 = cache[url2].copy()
    else:
        try:
            code = " ".join(url_to_code[url2].split())
        except Exception:
            code = ""
        code2 = tokenizer.tokenize(code)

    return convert_examples_to_features(code1, code2, label, url1, url2, tokenizer, args, cache)


class TextDataset(Dataset):

    def __init__(self, tokenizer, args, file_path: str = "train"):
        # e.g., Mutant_A_rip.csv -> Mutant_A_rip
        postfix = file_path.split("/")[-1].split(".csv")[0]
        self.examples = []
        self.args = args
        index_filename = file_path

        # load index
        logger.info("Creating features from index file at %s", index_filename)
        url_to_code = {}

        current_file_path = os.path.abspath(__file__)
        folder = os.path.join(os.path.dirname(current_file_path), "../cached_files")
        cache_file_path = os.path.join(folder, f"cached_{postfix}")
        code_pairs_file_path = os.path.join(folder, f"cached_{postfix}.pkl")
        code_pairs = []
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            # Try to load cached features directly
            self.examples = torch.load(cache_file_path)
            with open(code_pairs_file_path, "rb") as f:
                code_pairs = pickle.load(f)
            logger.info("Loading features from cached file %s", cache_file_path)
        except Exception:
            import csv

            logger.info("Creating features from dataset file at %s", file_path)
            # Read raw code from code_db_file
            with open(args.code_db_file, encoding="utf-8") as f:
                file_reader = csv.reader(f)
                file_header = next(file_reader)
                for line in file_reader:
                    url_to_code[line[0]] = extract_code_from_rip(line[1])

            # Build samples according to the index CSV
            data = []
            cache = {}
            with open(index_filename, encoding="utf-8") as f:
                file_reader = csv.reader(f)
                file_header = next(file_reader)
                for line in file_reader:
                    _, _, url1, url2, label = line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    if label == "0":
                        label = 0
                    else:
                        label = 1
                    data.append((url1, url2, label, tokenizer, args, cache, url_to_code))

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

            # convert examples to input features
            self.examples = [get_example(item) for item in data]
            torch.save(self.examples, cache_file_path)

        # Print the first three training examples for sanity check
        if "train" in postfix:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: %s", idx)
                logger.info("label: %s", example.label)
                logger.info(
                    "input_tokens: %s",
                    [x.replace("\\u0120", "_") for x in example.input_tokens],
                )
                logger.info("input_ids: %s", " ".join(map(str, example.input_ids)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            torch.tensor(self.examples[item].input_ids),
            torch.tensor(self.examples[item].label),
        )
