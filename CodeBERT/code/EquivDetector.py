import logging
import os
import time
import pickle
import random
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    PLBartConfig,
    PLBartForConditionalGeneration,
    PLBartTokenizer,
    RobertaForSequenceClassification,
)
from tqdm import tqdm

from CodeBERT.code.Model import Model
from CodeBERT.code.DatasetBuilder import TextDataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "plbart": (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    "codeT5-base": (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    "codebert-base": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "unixcoder-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "graphcodebert-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class EquivDetector:
    """Encapsulate training, validation, and testing pipeline for the equivalent mutant detection model."""

    def __init__(self, args):
        self.args = args

        # Setup CUDA, GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device

        logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

        # Set seed
        set_seed(args)

        # Build config / tokenizer / encoder model
        model_key = args.model_name_or_path.split("/")[-1]
        if model_key not in MODEL_CLASSES:
            raise ValueError(
                f"Unknown model type from model_name_or_path: {model_key}. "
                f"Expected one of {list(MODEL_CLASSES.keys())}."
            )
        config_class, encoder_class, tokenizer_class = MODEL_CLASSES[model_key]

        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path
        )
        config.num_labels = 2

        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
        encoder = encoder_class.from_pretrained(args.model_name_or_path)

        self.tokenizer = tokenizer
        self.model = Model(encoder, config, tokenizer, args)
        self.model.to(args.device)

        # Freeze or unfreeze encoder parameters according to requires_grad
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False if args.requires_grad == 0 else True

        logger.info("Training/evaluation parameters %s", args)

    # ---------- dataloader construction ----------

    def _build_train_dataloader(self, train_dataset: TextDataset) -> DataLoader:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=0,
        )
        return train_dataloader

    def _build_eval_dataloader(self, eval_dataset: TextDataset) -> DataLoader:
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            num_workers=0,
        )
        return eval_dataloader

    # ---------- training ----------

    def train(self) -> float:
        """Train the model; return best F1 on validation set."""
        args = self.args

        train_dataset = TextDataset(self.tokenizer, args, file_path=args.train_data_file)
        train_dataloader = self._build_train_dataloader(train_dataset)

        args.max_steps = args.epochs * len(train_dataloader)
        args.save_steps = len(train_dataloader)
        args.warmup_steps = args.max_steps // 5

        # optimizer & scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )

        # multi-gpu
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.epochs)
        logger.info(
            "  Instantaneous batch size per GPU = %d",
            args.train_batch_size // max(args.n_gpu, 1),
        )
        logger.info(
            "  Total train batch size = %d",
            args.train_batch_size * args.gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.max_steps)

        global_step = 0
        tr_loss, avg_loss, tr_num, train_loss = 0.0, 0.0, 0, 0.0
        best_f1 = 0.0
        self.model.zero_grad()

        for epoch in range(args.epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_num = 0
            train_loss = 0.0
            logger.info("-------------------------")
            for step, batch in enumerate(bar):
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                self.model.train()
                loss, logits, _ = self.model(inputs, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"epoch {epoch} loss {avg_loss}")

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    # Save once per epoch (equivalent to original save_steps = len(train_dataloader))
                    if global_step % args.save_steps == 0:
                        results, outputs_and_embeddings = self.evaluate()
                        # Save model checkpoint
                        if results["eval_f1"] > best_f1:
                            best_f1 = results["eval_f1"]
                            logger.info("  " + "*" * 20)
                            logger.info("  Best f1:%s", round(best_f1, 4))
                            logger.info("  " + "*" * 20)

                            checkpoint_prefix = "checkpoint-best-f1"
                            ckpt_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(ckpt_dir):
                                os.makedirs(ckpt_dir)
                            model_to_save = (
                                self.model.module if hasattr(self.model, "module") else self.model
                            )
                            output_model_file = os.path.join(ckpt_dir, "model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Saving model checkpoint to %s", output_model_file)

                            with open(
                                os.path.join(args.output_dir, "outputs_and_embeddings.pkl"),
                                "wb",
                            ) as f:
                                pickle.dump(outputs_and_embeddings, f)

        return best_f1

    @staticmethod
    def _compute_ece(prob_pos_list, ground_truth, n_bins: int = 10):
        """
        Compute Expected Calibration Error (ECE):
        - prob_pos_list: probability that each sample belongs to numeric label 1 (P(y=1))
        - ground_truth: corresponding true labels (0/1)
        - n_bins: number of bins
        Returns:
            ece (float) or None (if there is no valid sample)
        """
        valid = [
            (p, y)
            for p, y in zip(prob_pos_list, ground_truth)
            if p is not None and not np.isnan(p)
        ]
        if len(valid) == 0:
            return None

        probs = np.array([p for p, _ in valid], dtype=float)
        labels = np.array([y for _, y in valid], dtype=int)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        N = len(probs)

        for i in range(n_bins):
            left = bin_edges[i]
            right = bin_edges[i + 1]
            if i == 0:
                mask = (probs >= left) & (probs <= right)
            else:
                mask = (probs > left) & (probs <= right)

            if not np.any(mask):
                continue

            bin_probs = probs[mask]
            bin_labels = labels[mask]
            avg_conf = bin_probs.mean()
            avg_acc = (bin_labels == 1).mean()

            ece += (len(bin_probs) / N) * abs(avg_conf - avg_acc)

        return float(ece)

    # ---------- eval & test ----------

    def _run_eval(self, file_path: str, mode: str = "eval") -> Tuple[Dict[str, float], list]:
        args = self.args
        eval_dataset = TextDataset(self.tokenizer, args, file_path=file_path)
        eval_dataloader = self._build_eval_dataloader(eval_dataset)

        logger.info(f"***** Running {mode} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        logits = []
        y_trues = []
        embeddings = []

        T1 = time.perf_counter()
        for batch in tqdm(eval_dataloader):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit, embedding = self.model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
                embeddings.append(embedding.cpu().numpy())
            nb_eval_steps += 1
        
        num_pairs = len(eval_dataloader)
        T2 = time.perf_counter()
        print("Inference Time (per mutant pair): %s s" % ((T2 - T1) / num_pairs))
        logits = np.concatenate(logits, 0)
        y_trues = np.concatenate(y_trues, 0)
        embeddings = np.concatenate(embeddings, 0)
        best_threshold = 0.5
        
        prob_pos = logits[:, 1]
        ece = self._compute_ece(prob_pos, y_trues, n_bins=10)

        # Read cached code_pairs
        current_file_path = os.path.abspath(__file__)
        folder = os.path.join(os.path.dirname(current_file_path), "../cached_files")
        postfix = file_path.split("/")[-1].split(".csv")[0]
        code_pairs_file_path = os.path.join(folder, f"cached_{postfix}.pkl")
        with open(code_pairs_file_path, "rb") as f:
            code_pairs = np.array(pickle.load(f))[:, 2:]

        outputs_and_embeddings = [
            [
                code_pairs[i][0],
                code_pairs[i][1],
                y_trues[i],
                int(np.argmax(logits[i])),
                embeddings[i * 2],
                embeddings[i * 2 + 1],
            ]
            for i in range(len(y_trues))
        ]

        y_preds = logits[:, 1] > best_threshold
        recall = recall_score(y_trues, y_preds, average="macro", zero_division=0)
        precision = precision_score(y_trues, y_preds, average="macro", zero_division=0)
        f1 = f1_score(y_trues, y_preds, average="macro", zero_division=0)
        accuracy = accuracy_score(y_trues, y_preds)

        if mode == "eval":
            result = {
                "eval_accuracy": float(accuracy),
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_f1": float(f1),
                "eval_threshold": best_threshold,
                "eval_ece": float(ece),
            }
        else:
            result = {
                "test_accuracy": float(accuracy),
                "test_recall": float(recall),
                "test_precision": float(precision),
                "test_f1": float(f1),
                "test_ece": float(ece),
            }

        logger.info(f"***** {mode.capitalize()} results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        return result, outputs_and_embeddings

    def evaluate(self) -> Tuple[Dict[str, float], list]:
        """ Evaluate on validation set (eval_data_file). """
        return self._run_eval(self.args.eval_data_file, mode="eval")

    def test(self) -> Dict[str, float]:
        """ Evaluate on test set (test_data_file). """
        results, _ = self._run_eval(self.args.test_data_file, mode="test")
        return results
