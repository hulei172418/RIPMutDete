import os
import json
import pandas as pd
from tqdm import tqdm

from DeepseekCoder.code.LoadDataset import LoadDataset


class SFTDatasetBuilder:
    """
    Convert (code1, code2, label) in CSV into SFT JSONL files:
    - train_data_sft.jsonl
    - test_data_sft.jsonl
    """
    def __init__(self, codebase_data_file: str, train_data_file: str, test_data_file: str, output_dir: str):
        self.codebase_data_file = codebase_data_file
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.output_dir = output_dir

    def _build_single_split(self, dataset: LoadDataset, jsonl_path: str):
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx in tqdm(range(len(dataset)), desc=f"Building SFT dataset -> {os.path.basename(jsonl_path)}"):
                code_1, code_2, label = dataset[idx]
                instruction = "You are a Java mutation analysis assistant."
                content_prefix = (
                    "Two Java methods are given: the original version and its mutated version.\n"
                    "The Diff、JimpleChanges、content、Affect、CPG of the two java method are given: "
                    "the original version and the mutated version.\n"
                    "Your task is to determine if the two methods are semantically equivalent. "
                    "'Semantically equivalent' means: for any possible input, the two methods produce the "
                    "same outputs and have the same side effects.\n"
                    "Please only answer 'yes' or 'no'. 'yes' means they are semantically equal. "
                    "'no' means they are not.\n"
                )
                query = (
                    content_prefix
                    + "'''\n" + code_1 + "\n'''\n"
                    + "'''\n" + code_2 + "\n'''"
                )
                answer = "yes" if label == 1 else "no"
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]
                json.dump({"messages": messages}, f, ensure_ascii=False)
                f.write("\n")

    def build(self):
        train_dataset = LoadDataset(self.codebase_data_file, self.train_data_file)
        test_dataset = LoadDataset(self.codebase_data_file, self.test_data_file)

        os.makedirs(self.output_dir, exist_ok=True)
        train_jsonl = os.path.join(self.output_dir, "train_data_sft.jsonl")
        test_jsonl = os.path.join(self.output_dir, "eval_data_sft.jsonl")

        self._build_single_split(train_dataset, train_jsonl)
        self._build_single_split(test_dataset, test_jsonl)

        return train_jsonl, test_jsonl
