import pandas as pd
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    """Equivalent mutant dataset: retrieve code from codebase CSV and pair CSV using url1/url2"""
    def __init__(self, codebase_file: str, dataset_file: str):
        self.codebase = pd.read_csv(codebase_file)
        self.dataset = pd.read_csv(dataset_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        # Use url1/url2/label, matching the current CSV column names
        original_url = row["url1"]
        mutant_url = row["url2"]
        label = row["label"]

        code_1 = self.codebase.loc[self.codebase["url"] == original_url, "code"].values[0]
        code_2 = self.codebase.loc[self.codebase["url"] == mutant_url, "code"].values[0]
        return code_1, code_2, label

    def get_labels(self):
        return self.dataset["label"].tolist()

