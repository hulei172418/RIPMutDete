import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from GraphCodeBERT.code.DatasetBuilder import TextDataset

logger = logging.getLogger(__name__)


def set_seed(args):
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_ece(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
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


def train(args, train_dataset: TextDataset, model, tokenizer):
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

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
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0.0

    model.zero_grad()

    for epoch in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0.0

        for step, batch in enumerate(bar):
            (
                inputs_ids_1,
                position_idx_1,
                attn_mask_1,
                inputs_ids_2,
                position_idx_2,
                attn_mask_2,
                labels,
            ) = [x.to(args.device) for x in batch]

            model.train()
            loss, logits, _ = model(
                inputs_ids_1,
                position_idx_1,
                attn_mask_1,
                inputs_ids_2,
                position_idx_2,
                attn_mask_2,
                labels,
            )

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description(f"epoch {epoch} loss {avg_loss}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )

                if global_step % args.save_steps == 0:
                    results, outputs_and_embeddings = evaluate(args, model, tokenizer)

                    if results["eval_f1"] > best_f1:
                        best_f1 = results["eval_f1"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-f1"
                        output_dir = os.path.join(
                            args.output_dir, "..", f"{checkpoint_prefix}"
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        ckpt_path = os.path.join(output_dir, "model.bin")
                        torch.save(model_to_save.state_dict(), ckpt_path)
                        logger.info("Saving model checkpoint to %s", ckpt_path)

                        with open(
                            os.path.join(args.output_dir, "outputs_and_embeddings.pkl"),
                            "wb",
                        ) as f:
                            pickle.dump(outputs_and_embeddings, f)


def evaluate(args, model, tokenizer) -> Tuple[Dict[str, float], list]:
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    embeddings = []

    for batch in tqdm(eval_dataloader):
        (
            inputs_ids_1,
            position_idx_1,
            attn_mask_1,
            inputs_ids_2,
            position_idx_2,
            attn_mask_2,
            labels,
        ) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit, embedding = model(
                inputs_ids_1,
                position_idx_1,
                attn_mask_1,
                inputs_ids_2,
                position_idx_2,
                attn_mask_2,
                labels,
            )
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    embeddings = np.concatenate(embeddings, 0)
    best_threshold = 0.5
    
    prob_pos = logits[:, 1]
    ece = compute_ece(prob_pos, y_trues, n_bins=10)

    current_file_path = os.path.abspath(__file__)
    folder = os.path.join(os.path.dirname(current_file_path), "../cached_files")
    postfix = args.eval_data_file.split("/")[-1].split(".csv")[0]
    code_pairs_file_path = os.path.join(folder, f"cached_{postfix}.pkl")
    with open(code_pairs_file_path, "rb") as f:
        code_pairs = np.array(pickle.load(f))[:, 2:]
    outputs_and_embeddings = [
        [
            code_pairs[i][0],
            code_pairs[i][1],
            y_trues[i],
            np.argmax(logits[i]),
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

    result = {
        "eval_accuracy": float(accuracy),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
        "eval_ece": float(ece),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result, outputs_and_embeddings


def test(args, model, tokenizer, best_threshold: float = 0.5) -> Dict[str, float]:
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    for batch in tqdm(eval_dataloader):
        (
            inputs_ids_1,
            position_idx_1,
            attn_mask_1,
            inputs_ids_2,
            position_idx_2,
            attn_mask_2,
            labels,
        ) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit, _ = model(
                inputs_ids_1,
                position_idx_1,
                attn_mask_1,
                inputs_ids_2,
                position_idx_2,
                attn_mask_2,
                labels,
            )
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    
    prob_pos = logits[:, 1]
    ece = compute_ece(prob_pos, y_trues, n_bins=10)

    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds, average="macro", zero_division=0)
    precision = precision_score(y_trues, y_preds, average="macro", zero_division=0)
    f1 = f1_score(y_trues, y_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(y_trues, y_preds)

    result = {
        "test_accuracy": float(accuracy),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_ece": float(ece),
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result
