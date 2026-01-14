import argparse, os
from StarCoder.code.configs import TrainConfig
from StarCoder.code.EquivTrainer import EquivTrainer
from StarCoder.code.utils import ensure_dir, get_logger

logger = get_logger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="./StarCoder/models")
    p.add_argument("--tokenizer_name", type=str, default="./StarCoder/models")
    p.add_argument("--code_db_file", type=str, default="./dataset/Mutant_db_rip.csv")
    p.add_argument("--train_file", type=str, default="./dataset/Mutant_A_rip.csv")
    p.add_argument("--eval_file",  type=str, default="./dataset/Mutant_B_rip.csv")
    p.add_argument("--test_file",  type=str, default="./dataset/Mutant_C_rip.csv")
    p.add_argument("--output_dir", type=str, default="./StarCoder/saved_models/Equivalence")
    p.add_argument("--code_length", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size",  type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--requires_grad", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    cfg = TrainConfig(
        code_db_file=args.code_db_file,
        train_file=args.train_file,
        eval_file=args.eval_file,
        test_file=args.test_file,
        model_name_or_path=args.model_name_or_path,
        tokenizer_name=args.tokenizer_name,
        code_length=args.code_length,
        output_dir=args.output_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        requires_grad=args.requires_grad,
    )

    trainer = EquivTrainer(cfg)
    trainer.fit()
    metrics = trainer.test()
    logger.info("Final test metrics: %s", metrics)

if __name__ == "__main__":
    main()
