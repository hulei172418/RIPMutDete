#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatically construct a dataset from a spreadsheet (e.g., graph_1.xlsx) and a batch of output.json files
adapted for the CodeBERT equivalent mutant detection pipeline:

1. Read the table mutant_table (at minimum it should contain):
   - mutant_id:           unique mutant ID
   - mutant_graph_path:   directory containing JSON files; full path = mutant_graph_path + "/output.json"
   - label:               whether the mutant is equivalent (0/1 or mappable value)

   Optional:
   - split: defines train/eval/test split (values like 'train', 'eval', 'test')

2. Read JSON for each mutant and extract:
   - operator / Diff / SpecObserved / JimpleChanges
   - origin / mutated 下的 CODE / IR / Affected / Paths / CPG 信息

3. linearize this information into two text blocks (original/mutant)，写入新的 code_db_file:
      url,code

4. also generate three new index CSVs (train/eval/test) in the format:
      project_id,mutant_id,url1,url2,label
   so they can be fed into run.py directly (without requiring Mutant_A/B_hierarchical.csv)。

Dependencies:
    pip install pandas openpyxl
"""

import os
import json
import csv
import argparse
from typing import Dict, Any, List, Tuple

import random
import pandas as pd


class RIPDatasetBuilderAuto:
    """
     Builder that relies only on Excel + JSON, not the original Mutant_A/B CSV。
    """

    def __init__(
        self,
        mutant_table: str,
        out_code_db: str,
        out_train_csv: str,
        out_eval_csv: str,
        out_test_csv: str,
        mutant_id_col: str = "mutant_id",
        mutant_path_col: str = "mutant_graph_path",
        label_col: str = "label",
        split_col: str = None,
        train_ratio: float = 0.8,
        eval_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """
        Parameters:
        - mutant_table:    Path to the spreadsheet (e.g., ../dataset/graph_1.xlsx)
        - out_code_db:     Output path of the code_db_file
        - out_train_csv:   Output path of the training index CSV
        - out_eval_csv:    Output path of the validation index CSV
        - out_test_csv:    Output path of the test index CSV
        - mutant_id_col:   Column name of mutant id in the table (default "mutant_id")
        - mutant_path_col: Column name of the JSON path in the table (default "mutant_graph_path")
        - label_col:       Column name of the label (default "label")
        - split_col:       If the table already contains train/eval/test information, specify the column name
                          (e.g., "split"); otherwise random split will be used
        - train_ratio/eval_ratio/test_ratio: Proportions for random split (used if split_col is None)
        - random_seed:     Random seed used for splitting
        """
        self.mutant_table = mutant_table
        self.out_code_db = out_code_db
        self.out_train_csv = out_train_csv
        self.out_eval_csv = out_eval_csv
        self.out_test_csv = out_test_csv
        self.mutant_id_col = mutant_id_col
        self.mutant_path_col = mutant_path_col
        self.label_col = label_col
        self.split_col = split_col
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # {mutant_id: JSON object}
        self.json_by_mutant_id: Dict[str, Dict[str, Any]] = {}

    # -------------------- Utilities --------------------

    @staticmethod
    def safe_get(d: Dict[str, Any], path: List[str], default=None):
        cur = d
        for key in path:
            if not isinstance(cur, dict):
                return default
            if key not in cur:
                return default
            cur = cur[key]
        return cur if cur is not None else default

    @staticmethod
    def join_items(items, sep=" || "):
        if not items:
            return ""
        return sep.join(str(x) for x in items)

    @staticmethod
    def normalize_label(v) -> int:
        """
        Normalize various possible label values to 0/1.
        You can adjust the rules here according to your actual dataset.
        """
        if v is None:
            return 0
        s = str(v).strip().lower()

        # If numeric, treat as non-zero => 1
        try:
            num = float(s)
            return 1 if num != 0 else 0
        except Exception:
            pass

        # Common textual labels
        if s in ["1", "true", "yes", "y", "equivalent", "eqv", "non-equivalent", "neq"]:
            # Here we simply treat all as 1; you can distinguish equivalent/non-equivalent as needed
            return 1

        return 0

    # -------------------- Build "pseudo code text" from JSON --------------------

    def build_side_text(self, j: dict, side_label: str, side_key: str) -> str:
        """
        Linearize the JSON information for one side (origin / mutated) into a labeled text block for CodeBERT.

        side_label: "ORIGIN" or "MUT"
        side_key  : "origin" or "mutated"
        """
        lines: List[str] = []

        # ========= Top-level: mutant pair level =========
        op = self.safe_get(j, ["operator", "item"], "")
        if op:
            lines.append(f"[OP] {op}")

        diff = self.safe_get(j, ["Diff", "item"], "")
        if diff:
            lines.append(f"[DIFF] {diff}")

        domain = self.safe_get(j, ["DomainAssumptions", "items"], []) or []
        if domain:
            lines.append("[DOMAIN] " + self.join_items(domain, sep=" ; "))

        spec = self.safe_get(j, ["SpecObserved", "items"], []) or []
        if spec:
            lines.append("[SPEC] " + ", ".join(str(x) for x in spec))

        jimple_changes = self.safe_get(j, ["JimpleChanges", "items"], []) or []
        if jimple_changes:
            lines.append("[JIMPLE_CHANGES] " + self.join_items(jimple_changes))

        # ========= Information of the current side: origin / mutated =========
        side_obj = self.safe_get(j, [side_key, "item"], {}) or {}

        # Program id
        sid = self.safe_get(side_obj, ["id", "item"], "")
        if sid:
            lines.append(f"[{side_label}_ID] {sid}")

        # Source code
        code = self.safe_get(side_obj, ["content", "item"], "")
        if code:
            lines.append(f"[{side_label}_CODE]")
            lines.append(code)

        # Jimple IR
        ir = self.safe_get(side_obj, ["IR", "item"], "")
        if ir:
            lines.append(f"[{side_label}_IR]")
            lines.append(ir)

        # Affected statements
        affected = self.safe_get(side_obj, ["Affected", "items"], []) or []
        if affected:
            lines.append(f"[{side_label}_AFFECTED] " + self.join_items(affected))

        # Paths across the mutation point
        paths = self.safe_get(side_obj, ["Paths", "items"], []) or []
        if paths:
            lines.append(f"[{side_label}_PATHS] " + self.join_items(paths))

        # ========= CPG: CFG + DFG for each path =========
        cpg_items = self.safe_get(side_obj, ["CPG", "items"], []) or []

        def format_kv_item(obj) -> str:
            """Convert an object like {var:..., unit:..., sink:..., ...} into 'var=... | unit=... | ...'."""
            if not isinstance(obj, dict):
                return str(obj)
            parts = []
            for k, v in obj.items():
                if k == "comment":
                    continue
                parts.append(f"{k}={v}")
            return " | ".join(parts)

        for idx, cpg in enumerate(cpg_items):
            if not isinstance(cpg, dict):
                continue

            # ---- Path itself ----
            path = cpg.get("Path", "")
            if path:
                lines.append(f"[{side_label}_CPG_PATH_{idx}] {path}")

            # ---- CFG part: reachability + control dependencies ----
            cfg = cpg.get("CFG", {}) or {}

            dom_items = self.safe_get(cfg, ["dom", "items"], []) or []
            if dom_items:
                lines.append(
                    f"[{side_label}_CFG_DOM_{idx}] " + self.join_items(dom_items)
                )

            preds = self.safe_get(cfg, ["path_predicates", "items"], []) or []
            if preds:
                lines.append(
                    f"[{side_label}_CFG_PRED_{idx}] "
                    + " && ".join(str(p) for p in preds)
                )

            ctrl_deps = self.safe_get(cfg, ["control_deps_out", "items"], []) or []
            if ctrl_deps:
                lines.append(
                    f"[{side_label}_CFG_CTRL_DEPS_{idx}] "
                    + self.join_items(ctrl_deps)
                )

            # ---- DFG part: propagation info ----
            dfg = cpg.get("DFG", {}) or {}

            defs_at_mut = self.safe_get(dfg, ["defs_at_mut", "items"], []) or []
            uses_toward_output = self.safe_get(dfg, ["uses_toward_output", "items"], []) or []
            kill_set = self.safe_get(dfg, ["kill_set", "items"], []) or []
            heap_access = self.safe_get(dfg, ["heap_access", "items"], []) or []
            may_throw = self.safe_get(dfg, ["may_throw", "items"], []) or []
            alias_groups = self.safe_get(dfg, ["alias_groups", "items"], []) or []

            if defs_at_mut:
                lines.append(
                    f"[{side_label}_DFG_DEFS_AT_MUT_{idx}] "
                    + " || ".join(format_kv_item(x) for x in defs_at_mut)
                )

            if uses_toward_output:
                lines.append(
                    f"[{side_label}_DFG_USES_TOWARD_OUTPUT_{idx}] "
                    + " || ".join(format_kv_item(x) for x in uses_toward_output)
                )

            if kill_set:
                lines.append(
                    f"[{side_label}_DFG_KILL_SET_{idx}] "
                    + " || ".join(format_kv_item(x) for x in kill_set)
                )

            if heap_access:
                lines.append(
                    f"[{side_label}_DFG_HEAP_ACCESS_{idx}] "
                    + " || ".join(format_kv_item(x) for x in heap_access)
                )

            if may_throw:
                lines.append(
                    f"[{side_label}_DFG_MAY_THROW_{idx}] "
                    + " || ".join(format_kv_item(x) for x in may_throw)
                )

            if alias_groups:
                lines.append(
                    f"[{side_label}_DFG_ALIAS_GROUPS_{idx}] "
                    + " || ".join(format_kv_item(x) for x in alias_groups)
                )

            # Simple DFG summary
            if uses_toward_output or kill_set or heap_access or may_throw:
                summary_parts = [
                    f"uses={len(uses_toward_output)}",
                    f"killed={len(kill_set)}",
                    f"heap_access={len(heap_access)}",
                    f"may_throw={len(may_throw)}",
                ]
                lines.append(
                    f"[{side_label}_DFG_SUMMARY_{idx}] " + ", ".join(summary_parts)
                )

        return "\n".join(lines)

    # -------------------- Read table & JSON --------------------

    def load_json_by_mutant_id_from_table(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Read the table, load JSONs into self.json_by_mutant_id, and return the DataFrame and dictionary.
        """
        table_path = self.mutant_table
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Mutant table does not exist: {table_path}")

        df = pd.read_excel(table_path)

        for col in [self.mutant_id_col, self.mutant_path_col, self.label_col]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in the table; actual columns are:  {list(df.columns)}"
                )

        base_dir = os.path.dirname(table_path)
        result: Dict[str, Dict[str, Any]] = {}

        for idx, row in df.iterrows():
            mid = str(idx)
            path = str(row[self.mutant_path_col]).strip()

            if not mid or not path or path.lower() == "nan":
                continue

            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)

            if path.lower().endswith(".json"):
                json_path = path
            else:
                json_path = os.path.join(path, "output.json")

            if not os.path.exists(json_path):
                print(f"[WARN] JSON file does not exist: mutant_id={mid}, path={json_path}")
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read JSON: {json_path} ({e})")
                continue

            # Optional check: id inside JSON
            json_mid = self.safe_get(j, ["mutated", "item", "id", "item"], None)
            if json_mid is not None and str(json_mid).strip() != mid:
                print(f"[WARN] mutant_id in table={mid} is inconsistent with id in JSON={json_mid}, using table value")

            if mid in result:
                print(f"[WARN] Duplicate mutant_id={mid}; the later JSON will overwrite the former one.")

            result[mid] = j

        self.json_by_mutant_id = result
        print(f"[INFO] Loaded JSON for {len(result)} mutants from {table_path}.")

        return df, result

    # -------------------- Build code_db + index entries --------------------

    def build_entries(self, df: pd.DataFrame) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Build:
        - code_db_map: {url: code_text}
        - entries:     [{project_id, mutant_id, url1, url2, label, split}]
        based on the DataFrame and self.json_by_mutant_id.
        """
        code_db_map: Dict[str, str] = {}
        entries: List[Dict[str, Any]] = []

        # If there is a project column in the table, use it; otherwise use "P0"
        project_col = "project_id" if "project_id" in df.columns else None

        for idx, row in df.iterrows():
            mid = str(row[self.mutant_id_col]).strip()
            if mid not in self.json_by_mutant_id:
                continue

            label_raw = row[self.label_col]
            label = self.normalize_label(label_raw)

            if self.split_col and self.split_col in df.columns:
                split = str(row[self.split_col]).strip().lower()
                if split not in ["train", "eval", "valid", "validation", "test"]:
                    # Invalid split values are treated as train; you can modify this as needed
                    split = "train"
            else:
                split = None  # Random split will be applied later

            j = self.json_by_mutant_id[mid]

            # project_id
            project_id = str(row[project_col]).strip() if project_col else "P0"

            # Construct URLs
            new_url_origin = f"{mid}_orig"
            new_url_mut = f"{mid}_mut"

            # Construct text
            if new_url_origin not in code_db_map:
                code_db_map[new_url_origin] = self.build_side_text(j, "ORIGIN", "origin")

            if new_url_mut not in code_db_map:
                code_db_map[new_url_mut] = self.build_side_text(j, "MUT", "mutated")

            entry = {
                "project_id": project_id,
                "mutant_id": mid,
                "url1": new_url_origin,
                "url2": new_url_mut,
                "label": str(label),  # write as string in CSV
                "split": split,
            }
            entries.append(entry)

        print(f"[INFO] Constructed {len(entries)} mutant records.")
        return code_db_map, entries

    # -------------------- Split into train/eval/test --------------------

    def split_entries(
        self, entries: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return train_entries, eval_entries, test_entries.
        """
        # If the table already has split information, split by that
        if any(e["split"] for e in entries):
            train, eval_, test = [], [], []
            for e in entries:
                s = (e["split"] or "").lower()
                if s in ["train"]:
                    train.append(e)
                elif s in ["eval", "valid", "validation"]:
                    eval_.append(e)
                elif s in ["test"]:
                    test.append(e)
                else:
                    train.append(e)
            print(
                f"[INFO] Split by table column: train={len(train)}, eval={len(eval_)}, test={len(test)}"
            )
            return train, eval_, test

        # Otherwise, perform random split
        random.seed(self.random_seed)
        shuffled = entries[:]
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_eval = int(n * self.eval_ratio)

        train = shuffled[:n_train]
        eval_ = shuffled[n_train:n_train + n_eval]
        test = shuffled[n_train + n_eval:]

        print(
            f"[INFO] Random split: train={len(train)}, eval={len(eval_)}, test={len(test)} "
            f"(ratio={self.train_ratio}/{self.eval_ratio}/{self.test_ratio})"
        )
        return train, eval_, test

    # -------------------- Write CSV / code_db --------------------

    @staticmethod
    def write_index_csv(path: str, entries: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["project_id", "mutant_id", "url1", "url2", "label"])
            for e in entries:
                writer.writerow(
                    [
                        e["project_id"],
                        e["mutant_id"],
                        e["url1"],
                        e["url2"],
                        e["label"],
                    ]
                )

    @staticmethod
    def write_code_db(path: str, code_db_map: Dict[str, str]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "code"])
            for url, code in code_db_map.items():
                writer.writerow([url, code])

    # -------------------- Public entrypoint --------------------

    def run(self) -> None:
        """
        Overall pipeline:
        1. Read JSON from the table
        2. Construct code_db_map & entries
        3. Split into train/eval/test
        4. Write out code_db and the three index CSVs
        """
        df, _ = self.load_json_by_mutant_id_from_table()
        code_db_map, entries = self.build_entries(df)
        train_entries, eval_entries, test_entries = self.split_entries(entries)

        self.write_code_db(self.out_code_db, code_db_map)
        self.write_index_csv(self.out_train_csv, train_entries)
        self.write_index_csv(self.out_eval_csv, eval_entries)
        self.write_index_csv(self.out_test_csv, test_entries)

        print("[INFO] All files have been generated:")
        print(f"  code_db_file = {self.out_code_db}")
        print(f"  train_csv    = {self.out_train_csv}")
        print(f"  eval_csv     = {self.out_eval_csv}")
        print(f"  test_csv     = {self.out_test_csv}")


# ============================ CLI entrypoint ============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically build a CodeBERT equivalent mutant dataset from table + JSON (no original Mutant_A/B CSV required)"
    )
    parser.add_argument(
        "--mutant_table",
        default="./dataset/graph_1.xlsx",
        help="Spreadsheet containing mutant_id / mutant_graph_path / label, e.g., ../dataset/graph_1.xlsx",
    )
    parser.add_argument(
        "--out_code_db",
        default="./dataset/Mutant_db_rip.csv",
        help="Output path of code_db_file, e.g., ../dataset/Mutant_db_rip.csv",
    )
    parser.add_argument(
        "--out_train_csv",
        default="./dataset/Mutant_A_rip.csv",
        help="Output path of training CSV, e.g., ../dataset/Mutant_A_rip.csv",
    )
    parser.add_argument(
        "--out_eval_csv",
        default="./dataset/Mutant_B_rip.csv",
        help="Output path of validation CSV, e.g., ../dataset/Mutant_B_rip.csv",
    )
    parser.add_argument(
        "--out_test_csv",
        default="./dataset/Mutant_C_rip.csv",
        help="Output path of test CSV, e.g., ../dataset/Mutant_C_rip.csv",
    )
    parser.add_argument(
        "--mutant_id_col",
        default="mutant_id_col",
        help="Column name for mutant id in the table, default mutant_id",
    )
    parser.add_argument(
        "--mutant_path_col",
        default="mutant_graph_path",
        help="Column name for JSON path in the table, default mutant_graph_path",
    )
    parser.add_argument(
        "--label_col",
        default="label",
        help="Column name for labels (equivalent/non-equivalent), default label",
    )
    parser.add_argument(
        "--split_col",
        default=None,
        help="Column name indicating train/eval/test in the table (if any), e.g., split; if None, random split is used",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training ratio when using random split (effective when split_col is None)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Validation ratio when using random split",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio when using random split",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    builder = RIPDatasetBuilderAuto(
        mutant_table=args.mutant_table,
        out_code_db=args.out_code_db,
        out_train_csv=args.out_train_csv,
        out_eval_csv=args.out_eval_csv,
        out_test_csv=args.out_test_csv,
        mutant_id_col=args.mutant_id_col,
        mutant_path_col=args.mutant_path_col,
        label_col=args.label_col,
        split_col=args.split_col,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )
    builder.run()


if __name__ == "__main__":
    main()
