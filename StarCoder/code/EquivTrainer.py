import os
import numpy as np
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from StarCoder.code.configs import TrainConfig
from StarCoder.code.datasets import CodePairDataset
from StarCoder.code.modules import PairClassifier
from StarCoder.code.utils import get_logger, set_seed, ensure_dir

logger = get_logger(__name__)


class EquivTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        tok_name = cfg.tokenizer_name or cfg.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_cfg = AutoConfig.from_pretrained(cfg.model_name_or_path)
        base_cfg.num_labels = cfg.num_labels
        self.encoder = AutoModel.from_pretrained(cfg.model_name_or_path, config=base_cfg)

        hidden = base_cfg.hidden_size
        dropout = getattr(base_cfg, "hidden_dropout_prob", 0.1)
        self.model = PairClassifier(
            encoder=self.encoder,
            hidden_size=hidden,
            dropout=dropout,
            num_labels=cfg.num_labels,
            code_length=cfg.code_length,
            pad_token_id=self.tokenizer.pad_token_id,
        ).to(cfg.device)

        if cfg.requires_grad == 0:
            for n, p in self.model.named_parameters():
                if "classifier" not in n:
                    p.requires_grad = False

    @staticmethod
    def _compute_ece(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        assert prob_pos.shape[0] == y_true.shape[0]
        N = prob_pos.shape[0]
        if N == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            start = bin_edges[i]
            end = bin_edges[i + 1]

            if i == 0:
                mask = (prob_pos >= start) & (prob_pos <= end)
            else:
                mask = (prob_pos > start) & (prob_pos <= end)

            if not np.any(mask):
                continue

            bin_probs = prob_pos[mask]
            bin_labels = y_true[mask]

            avg_conf = bin_probs.mean()
            avg_acc = (bin_labels == 1).mean()

            ece += (bin_probs.shape[0] / N) * abs(avg_conf - avg_acc)

        return float(ece)

    # -------- dataset builders --------
    def _build_loader(self, csv_path: str, train: bool) -> DataLoader:
        ds = CodePairDataset(self.tokenizer, self.cfg.code_length, self.cfg.code_db_file, csv_path)
        sampler = RandomSampler(ds) if train else SequentialSampler(ds)
        bs = self.cfg.train_batch_size if train else self.cfg.eval_batch_size
        return DataLoader(ds, sampler=sampler, batch_size=bs, pin_memory=True)

    # -------- training / eval / test --------
    def fit(self):
        train_loader = self._build_loader(self.cfg.train_file, train=True)

        no_decay = ["bias", "LayerNorm.weight"]
        grouped = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": self.cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(grouped, lr=self.cfg.learning_rate, eps=self.cfg.adam_epsilon)

        max_steps = self.cfg.epochs * len(train_loader)
        warmup = max(1, max_steps // 5)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max_steps)

        global_step, best_f1 = 0, 0.0
        self.model.zero_grad()
        logger.info("***** Training *****")
        logger.info("  #examples=%d  epochs=%d", len(train_loader.dataset), self.cfg.epochs)

        for ep in range(self.cfg.epochs):
            self.model.train()
            running, seen = 0.0, 0
            for step, batch in enumerate(train_loader):
                input_ids, labels = (t.to(self.cfg.device) for t in batch)
                loss, _, _ = self.model(input_ids, labels)

                if self.cfg.gradient_accumulation_steps > 1:
                    loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    optimizer.step(); scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                running += loss.item(); seen += 1
                if (step + 1) % 10 == 0:
                    logger.info("epoch %d step %d/%d  loss=%.5f", ep, step+1, len(train_loader), running/seen)

            metrics, _ = self.evaluate(self.cfg.eval_file)
            if metrics["eval_f1"] > best_f1:
                best_f1 = metrics["eval_f1"]
                self._save_best()

    def evaluate(self, eval_csv: str) -> Tuple[Dict[str, float], np.ndarray]:
        loader = self._build_loader(eval_csv, train=False)
        self.model.eval()
        eval_loss, logits, y_true, embs = 0.0, [], [], []

        with torch.no_grad():
            for batch in loader:
                input_ids, labels = (t.to(self.cfg.device) for t in batch)
                loss, logit, emb = self.model(input_ids, labels)
                eval_loss += loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_true.append(labels.cpu().numpy())
                embs.append(emb.cpu().numpy())

        logits = np.concatenate(logits, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        embs   = np.concatenate(embs, axis=0)
        
        prob_pos = logits[:, 1]
        eval_ece = self._compute_ece(prob_pos, y_true, n_bins=10)

        y_pred = (logits[:, 1] > 0.5)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        result = {
            "eval_accuracy": float(accuracy), 
            "eval_precision": float(precision), 
            "eval_recall": float(recall), 
            "eval_f1": float(f1),
            "eval_ece": float(eval_ece),
            }
        logger.info("Eval: %s", {k: round(v, 4) for k, v in result.items()})
        return result, embs

    def test(self) -> Dict[str, float]:
        loader = self._build_loader(self.cfg.test_file, train=False)
        self.model.eval()
        logits, y_true = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids, labels = (t.to(self.cfg.device) for t in batch)
                _, logit, _ = self.model(input_ids, labels)
                logits.append(logit.cpu().numpy())
                y_true.append(labels.cpu().numpy())

        logits = np.concatenate(logits, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = (logits[:, 1] > 0.5)
        
        prob_pos = logits[:, 1]
        test_ece = self._compute_ece(prob_pos, y_true, n_bins=10)

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        result = {
            "eval_accuracy": float(accuracy), 
            "test_precision": float(precision), 
            "test_recall": float(recall), 
            "test_f1": float(f1),
            "test_ece": float(test_ece),
            }
        logger.info("Test: %s", {k: round(v, 4) for k, v in result.items()})
        return result

    # -------- save best --------
    def _save_best(self):
        ckpt_dir = os.path.join(self.cfg.output_dir, "checkpoint-best-f1")
        ensure_dir(ckpt_dir)
        path = os.path.join(ckpt_dir, "model.bin")
        torch.save(self.model.state_dict(), path)
        logger.info("Saved best to %s", path)
