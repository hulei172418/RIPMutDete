import argparse
import logging
import os
import torch

from transformers import (
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

from GraphCodeBERT.code.Model import Model
from GraphCodeBERT.code.DatasetBuilder import TextDataset, InputFeatures
from GraphCodeBERT.code.EquivDetector import set_seed, train, evaluate, test
from GraphCodeBERT.code.utils import get_max_position_embeddings

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "plbart-base": (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer),
    "codet5-base": (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    "codebert-base": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "unixcoder-base": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
    ),
    "graphcodebert-base": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
    ),
}


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--code_db_file",
        default="./dataset/Mutant_db_rip.csv",
        type=str,
        help="CSV file containing all code snippets (url, code).",
    )
    parser.add_argument(
        "--requires_grad",
        default=0,
        type=int,
        help="If 0, freeze encoder parameters and only train classifier head.",
    )
    parser.add_argument(
        "--train_data_file",
        default="./dataset/Mutant_A_rip.csv",
        type=str,
        help="The input training data file (CSV with mutant pairs).",
    )
    parser.add_argument(
        "--output_dir",
        default="./GraphCodeBERT/saved_models/Equivalence",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--eval_data_file",
        default="./dataset/Mutant_B_rip.csv",
        type=str,
        help="Evaluation CSV file.",
    )
    parser.add_argument(
        "--test_data_file",
        default="./dataset/Mutant_C_rip.csv",
        type=str,
        help="Test CSV file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="./GraphCodeBERT/graphcodebert-base",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--config_name",
        default="./GraphCodeBERT/graphcodebert-base",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="./GraphCodeBERT/graphcodebert-base",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path.",
    )

    parser.add_argument(
        "--code_length",
        default=384,
        type=int,
        help="Code input sequence length after tokenization.",
    )
    parser.add_argument(
        "--data_flow_length",
        default=128,
        type=int,
        help="Optional Data Flow input sequence length after tokenization.",
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the dev set.",
    )
    parser.add_argument(
        "--do_test",
        default=True,
        action="store_true",
        help="Whether to run eval on the test set.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Random seed for initialization.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs.",
    )

    args = parser.parse_args()
    return args


def main():
    args = build_args()
    
    config_path = os.path.join(args.model_name_or_path, "config.json")
    args.code_length = get_max_position_embeddings(config_path, default_value=514) - args.data_flow_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args)

    key = args.model_name_or_path.split("/")[-1]
    config_class, model_class, tokenizer_class = MODEL_CLASSES[key]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    encoder = model_class.from_pretrained(args.model_name_or_path)

    model = Model(encoder, config, tokenizer, args)
    model.to(args.device)

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False if args.requires_grad == 0 else True

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    results = {}
    if args.do_eval:
        checkpoint_prefix = "checkpoint-best-f1/model.bin"
        ckpt_path = os.path.join(args.output_dir, "..", checkpoint_prefix)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.to(args.device)
        results, _ = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = "checkpoint-best-f1/model.bin"
        ckpt_path = os.path.join(args.output_dir, "..", checkpoint_prefix)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
        model.to(args.device)
        results = test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
