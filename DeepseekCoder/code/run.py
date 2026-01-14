import os
import argparse


from DeepseekCoder.code.Application import Application
from DeepseekCoder.code.utils.util import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--codebase_data_file", default="./dataset/Mutant_db_rip.csv", type=str,
                        help="The code db data file (a csv file).")
    parser.add_argument("--train_data_file", default="./dataset/Mutant_A_rip.csv", type=str,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default="./DeepseekCoder/saved_models/Equivalence", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--eval_data_file", default="./dataset/Mutant_B_rip.csv", type=str,
                        help="Optional eval data file (unused in current code).")
    parser.add_argument("--test_data_file", default="./dataset/Mutant_C_rip.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a csv file).")
    parser.add_argument("--test_type", default="zero-shot-prompt", type=str,
                        help="Three types of testing: zero-shot-prompt, few-shot-prompt, and inference_from_ckpt")
    parser.add_argument("--model_type", default="llama", type=str,
                        help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--model_name_or_path", default="./DeepseekCoder/models", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_dir", default="./DeepseekCoder/saved_models/checkpoints/",
                        type=str, help="The path of trained checkpoints for inference from that ckpt")
    parser.add_argument("--tokenizer_name", default="./DeepseekCoder/models", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pretrained models downloaded from s3")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="(Unused) Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", default=True,
                        help="Whether to run test.")

    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before backward/update.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints with the same prefix as model_name_or_path.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--epoch", type=int, default=10,
                        help="Number of epochs for training model (unused)")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    app = Application(args)

    if args.do_train:
        app.run_train()

    if args.do_test:
        app.run_test()


if __name__ == "__main__":
    main()
