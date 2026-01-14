import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from GPT_3_5.code.ChatEngine import ChatEngine
from GPT_3_5.code.LoadDataset import LoadDataset
from GPT_3_5.code.utils.util import truncate_code_to_tokens


class Tester:
    """
    Encapsulates various test modes:
    - zero-shot-prompt
    - few-shot-prompt
    - eval_after_ft (shares logic with zero-shot, only differs in output file name)
    - others (inference_from_ckpt)
    """
    def __init__(self, args, chat_engine: ChatEngine):
        self.args = args
        self.chat_engine = chat_engine
        
    @staticmethod
    def _compute_ece(prob_pos_list, ground_truth, n_bins: int = 10):
        valid_pairs = []
        for p, gt in zip(prob_pos_list, ground_truth):
            if p is None:
                continue
            try:
                if np.isnan(p):
                    continue
            except TypeError:
                pass
            valid_pairs.append((float(p), int(gt)))

        if not valid_pairs:
            return None

        probs = np.array([p for p, _ in valid_pairs], dtype=float)
        labels = np.array([gt for _, gt in valid_pairs], dtype=int)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(probs)

        for i in range(n_bins):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            if i < n_bins - 1:
                mask = (probs >= left) & (probs < right)
            else:
                mask = (probs >= left) & (probs <= right)

            if not np.any(mask):
                continue

            bin_probs = probs[mask]
            bin_labels = labels[mask]

            bin_conf = np.mean(bin_probs)
            bin_acc = np.mean(bin_labels == 1)

            ece += np.abs(bin_acc - bin_conf) * (len(bin_probs) / n_total)

        return float(ece)

    @staticmethod
    def _parse_predictions(messages):
        prediction_list = []
        unrecognized_count = 0
        for item in messages:
            low = item.lower()
            if "yes" in low and "no" not in low:
                prediction_list.append(1)
            elif "no" in low and "yes" not in low:
                prediction_list.append(0)
            else:
                prediction_list.append(-1)
                print("[Unrecognized LLM Output]:", item)
                unrecognized_count += 1
        return prediction_list, unrecognized_count

    @staticmethod
    def _compute_metrics(ground_truth, prediction_list):
        valid_pairs = [(gt, pred) for gt, pred in zip(ground_truth, prediction_list) if pred != -1]
        filtered_ground_truth = [gt for gt, _ in valid_pairs]
        filtered_prediction = [pred for _, pred in valid_pairs]

        precision, recall, fscore, _ = precision_recall_fscore_support(
            filtered_ground_truth, filtered_prediction, average="macro"
        )
        accuracy = accuracy_score(filtered_ground_truth, filtered_prediction)
        return accuracy, precision, recall, fscore

    @staticmethod
    def load_fewshot_example(base_dir: str, idx: int):
        """
        return:
            exp_code_1 (str), exp_code_2 (str), explanation (str)
        """
        fewshot_dir = os.path.join(base_dir, "../FewShotExample")

        file_path_1 = os.path.abspath(os.path.join(fewshot_dir, f"exp_code_{idx}_1.txt"))
        file_path_2 = os.path.abspath(os.path.join(fewshot_dir, f"exp_code_{idx}_2.txt"))
        explanation_path = os.path.abspath(os.path.join(fewshot_dir, f"explaination_{idx}.txt"))

        with open(file_path_1, encoding="utf8") as f:
            exp_code_1 = f.read()
        with open(file_path_2, encoding="utf8") as f:
            exp_code_2 = f.read()
        with open(explanation_path, encoding="utf8") as f:
            explanation = f.read()

        return exp_code_1, exp_code_2, explanation

    def _save_results(self, messages_list, prediction_list, output_dir, filename_prefix):
        result_dict = {"Message": messages_list, "Numerical_Pred": prediction_list}
        result_df = pd.DataFrame.from_dict(result_dict)
        os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(os.path.join(output_dir, f"{filename_prefix}.csv"), index=False)

    def _run_zero_shot_or_eval_after_ft(self, codebase_data, test_data, output_dir, test_type):
        dataset = LoadDataset(codebase_data, test_data)
        messages_list = []
        max_context = self.chat_engine.max_tokens
        max_code_tokens = max_context // 4  # Each code snippet can use at most 1/4 context length

        for idx in tqdm(range(len(dataset)), desc=f"Testing ({test_type})"):
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

            content = (
                content_prefix
                + "'''\n"
                + code_1
                + "\n'''\n"
                + "'''\n"
                + code_2
                + "\n'''"
            )

            message = self.chat_engine.chat(instruction, content)
            messages_list.append(message)

        prediction_list, unrecognized_count = self._parse_predictions(messages_list)
        ground_truth = dataset.get_labels()
        
        prob_equiv_list = []
        for pred in prediction_list:
            if pred == -1:
                prob_equiv_list.append(None)
            else:
                prob_equiv_list.append(float(pred))
        
        ece = self._compute_ece(prob_equiv_list, ground_truth, n_bins=10)
        accuracy, precision, recall, fscore = self._compute_metrics(ground_truth, prediction_list)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {fscore}")
        print(f"Unrecognized outputs: {unrecognized_count}/{len(dataset)}")
        if ece is not None:
            print(f"ECE: {ece}")
        else:
            print("ECE: N/A (no valid probability scores)")
        prefix = f"ZeroShotResult" if test_type == "zero-shot-prompt" else f"FineTuningResult"
        self._save_results(messages_list, prediction_list, output_dir, prefix)
        return len(dataset)

    def _run_few_shot(self, codebase_data, test_data, output_dir):
        example_candidates = LoadDataset(codebase_data, self.args.train_data_file)
        dataset = LoadDataset(codebase_data, test_data)
        max_context = self.chat_engine.max_tokens
        max_code_tokens = max_context // 4
        pos_examples, neg_examples = [], []

        for i in range(len(example_candidates)):
            c1, c2, label = example_candidates[i]
            if label == 1:
                pos_examples.append(example_candidates[i])
            else:
                neg_examples.append(example_candidates[i])

        exp_code_0_1, exp_code_0_2, explaination_0 = self.load_fewshot_example(self.args.train_data_file, 0)
        exp_code_1_1, exp_code_1_2, explaination_1 = self.load_fewshot_example(self.args.train_data_file, 1)
        exp_code_2_1, exp_code_2_2, explaination_2 = self.load_fewshot_example(self.args.train_data_file, 2)
        example_0 = (
            "Here are two semantically equal and a not equivalent examples : \n"
            "The first example pair is \n"
            "'''\n"
            + exp_code_0_1
            + "\n'''\n"
            "'''\n"
            + exp_code_0_2
            + "\n'''\n"
            + explaination_0
            + "\n"
        )
        example_1 = (
            "The second example pair is \n"
            "''' \n"
            + exp_code_1_1
            + "\n'''\n"
            "'''\n"
            + exp_code_1_2
            + "\n'''\n"
            + explaination_1
            + "\n"
        )
        example_2 = (
            "The third example pair is \n"
            "'''\n"
            + exp_code_2_1
            + "\n'''\n"
            "'''\n"
            + exp_code_2_2
            + "\n'''\n"
            + explaination_2
        )
            
        messages_list = []

        for idx in tqdm(range(len(dataset)), desc="Testing (few-shot)"):
            code_1, code_2, label = dataset[idx]

            instruction = "You are a Java mutation analysis assistant."
            content_prefix = (
                "Two Java methods are given: the original version and its mutated version.\n"
                "Please only answer 'yes' or 'no'. 'yes' means they are semantically equal. "
                "'no' means they are not.\n"
            )
            content = (
                content_prefix
                + "'''\n"
                + code_1
                + "\n'''\n"
                + "'''\n"
                + code_2
                + "\n'''"
            )
            
            # print("The token is: ", token_count)
            message = self.chat_engine.chat(instruction + example_0 + example_1 + example_2, content)
            messages_list.append(message)

        prediction_list, unrecognized_count = self._parse_predictions(messages_list)
        ground_truth = dataset.get_labels()
        
        prob_equiv_list = []
        for pred in prediction_list:
            if pred == -1:
                prob_equiv_list.append(None)
            else:
                prob_equiv_list.append(float(pred))
        ece = self._compute_ece(prob_equiv_list, ground_truth, n_bins=10)
        accuracy, precision, recall, fscore = self._compute_metrics(ground_truth, prediction_list)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {fscore}")
        print(f"Unrecognized outputs: {unrecognized_count}/{len(dataset)}")
        if ece is not None:
            print(f"ECE: {ece}")
        else:
            print("ECE: N/A (no valid probability scores)")
        self._save_results(messages_list, prediction_list, output_dir, f"FewShotResult")
        return len(dataset)

    def _run_other_eval(self, codebase_data, test_data, output_dir):
        dataset = LoadDataset(codebase_data, test_data)
        max_context = self.chat_engine.max_tokens
        max_code_tokens = max_context // 4
        messages_list = []

        for idx in tqdm(range(len(dataset)), desc="Testing (eval_from_ckpt)"):
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


            content = (
                content_prefix
                + "'''\n"
                + code_1
                + "\n'''\n"
                + "'''\n"
                + code_2
                + "\n'''"
            )
            
            message = self.chat_engine.chat(instruction, content)
            messages_list.append(message)

        prediction_list, unrecognized_count = self._parse_predictions(messages_list)
        ground_truth = dataset.get_labels()
        
        prob_equiv_list = []
        for pred in prediction_list:
            if pred == -1:
                prob_equiv_list.append(None)
            else:
                prob_equiv_list.append(float(pred))
        ece = self._compute_ece(prob_equiv_list, ground_truth, n_bins=10)
        accuracy, precision, recall, fscore = self._compute_metrics(ground_truth, prediction_list)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {fscore}")
        print(f"Unrecognized outputs: {unrecognized_count}/{len(dataset)}")
        if ece is not None:
            print(f"ECE: {ece}")
        else:
            print("ECE: N/A (no valid probability scores)")
        self._save_results(messages_list, prediction_list, output_dir, f"EvalResult")
        
        return len(dataset)

    def run(self, test_type: str, codebase_data: str, test_data: str, output_dir: str) -> int:
        if test_type in ["zero-shot-prompt", "eval_after_ft"]:
            return self._run_zero_shot_or_eval_after_ft(codebase_data, test_data, output_dir, test_type)
        elif test_type == "few-shot-prompt":
            return self._run_few_shot(codebase_data, test_data, output_dir)
        else:
            return self._run_other_eval(codebase_data, test_data, output_dir)
