from typing import List, Tuple, Dict
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from StarCoder.code.utils import get_logger, ensure_dir


logger = get_logger(__name__)

class CodePairDataset(Dataset):
    """
    Output: (input_ids, label)
    - Each sample = two code snippets, each truncated/padded to `code_length`,
      then concatenated into a 2 * code_length sequence of ids.
    - Supports two types of input CSV:
      1) Direct columns: ['code_1', 'code_2', 'label']
      2) URL-based indexing:
         - pair_csv: the 3rd/4th/5th columns are url1, url2, label
         - code_db_csv: the first two columns are url, code
    """
    def __init__(self, tokenizer, code_length: int, code_db_csv: str, pair_csv: str):
        self.tokenizer = tokenizer
        self.code_length = code_length
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_files")
        ensure_dir(cache_dir)
        tag = os.path.basename(pair_csv).split(".csv")[0]
        cache_path = os.path.join(cache_dir, f"cached_{tag}.pt")

        if os.path.exists(cache_path):
            logger.info("Loading cached features: %s", cache_path)
            data = torch.load(cache_path)
            self.examples = data["examples"]
            return

        logger.info("Building features from raw: %s", pair_csv)
        df_pairs = pd.read_csv(pair_csv)

        if {"code_1", "code_2", "label"}.issubset(set(df_pairs.columns)):
            for _, row in df_pairs.iterrows():
                c1 = str(row["code_1"]); c2 = str(row["code_2"]); lab = int(row["label"])
                ids = self._encode_pair(c1, c2)
                self.examples.append((ids, torch.tensor(lab)))
        else:
            df_db = pd.read_csv(code_db_csv)
            url_to_code: Dict[str, str] = dict(zip(df_db.iloc[:, 0].astype(str), df_db.iloc[:, 1].astype(str)))

            for _, row in df_pairs.iterrows():
                url1 = str(row.iloc[2]); url2 = str(row.iloc[3]); lab = int(row.iloc[4])
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                c1 = url_to_code[url1]; c2 = url_to_code[url2]
                ids = self._encode_pair(c1, c2)
                self.examples.append((ids, torch.tensor(lab)))

        torch.save({"examples": self.examples}, cache_path)
        logger.info("Saved cache: %s | num_examples=%d", cache_path, len(self.examples))

    def _encode_one(self, code: str) -> List[int]:
        pad_id = self.tokenizer.pad_token_id
        ids = self.tokenizer.encode(
            code,
            add_special_tokens=True,
            truncation=True,
            max_length=self.code_length,
        )
        if len(ids) < self.code_length:
            ids = ids + [pad_id] * (self.code_length - len(ids))
        else:
            ids = ids[: self.code_length]
        return ids

    def _encode_pair(self, code1: str, code2: str) -> torch.Tensor:
        ids1 = self._encode_one(code1)
        ids2 = self._encode_one(code2)
        all_ids = ids1 + ids2
        return torch.tensor(all_ids, dtype=torch.long)

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]
