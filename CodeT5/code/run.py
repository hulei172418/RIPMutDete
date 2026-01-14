import argparse
import logging
import os

from CodeT5.code.EquivDetector import EquivDetector

logger = logging.getLogger(__name__)


def build_args():
    parser = argparse.ArgumentParser()

    # Required parameters
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
        default="./CodeT5/saved_models/Equivalence",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
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
        default="./CodeT5/codeT5-base",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--config_name",
        default="./CodeT5/codeT5-base",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="./CodeT5/codeT5-base",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path.",
    )

    parser.add_argument(
        "--code_length",
        default=512,
        type=int,
        help="Code input sequence length after tokenization.",
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
        action="store_true",
        default=True,
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

    # make sure output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Starting equivalence detection pipeline.")

    args = build_args()
    detector = EquivDetector(args)

    results = {}
    if args.do_train:
        best_f1 = detector.train()
        logger.info("Training finished. Best eval F1: %.4f", best_f1)
    if args.do_eval:
        results, _ = detector.evaluate()
    if args.do_test:
        results = detector.test()

    logger.info("Final results: %s", results)
    return results


if __name__ == "__main__":
    main()