from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    # Data
    code_db_file: str = "./dataset/Mutant_db_rip.csv"
    train_file: str = "./dataset/Mutant_A_rip.csv"
    eval_file:  str = "./dataset/Mutant_B_rip.csv"
    test_file:  str = "./dataset/Mutant_C_rip.csv"

    # Model and tokenizer
    model_name_or_path: str = "./StarCoder/models"
    tokenizer_name: str = "./StarCoder/models"
    code_length: int = 512
    num_labels: int = 2
    requires_grad: int = 0

    # Training
    output_dir: str = "./StarCoder/saved_models/Equivalence"
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    train_batch_size: int = 1
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Randomness and device
    seed: int = 123456
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
