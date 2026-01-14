import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

from QwenCoder.code.ChatEngine import ChatEngine
from QwenCoder.code.Tester import Tester
from QwenCoder.code.LoRATrainer import LoRATrainer
from QwenCoder.code.SFTDatasetBuilder import SFTDatasetBuilder
from QwenCoder.code.utils.util import *

class Application:
    """
    Overall application wrapper:
    - Parse arguments
    - Construct SFT dataset
    - LoRA training
    - Various testing modes
    """
    def __init__(self, args):
        self.args = args
        config_path = os.path.join(args.model_name_or_path, "config.json")
        self.max_tokens = get_max_position_embeddings(config_path, default_value=4096)
        self.n_gpus = torch.cuda.device_count()
        self.max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory/ (1024 ** 3)*0.9:.2f}GiB" \
            for i in range(self.n_gpus)} if self.n_gpus > 0 else None
        self.quant_config = BitsAndBytesConfig(load_in_8bit=True)

    def run_train(self):
        # Build SFT dataset
        builder = SFTDatasetBuilder(
            self.args.codebase_data_file,
            self.args.train_data_file,
            self.args.eval_data_file,
            self.args.output_dir,
        )
        train_jsonl, eval_jsonl = builder.build()

        # Load SFT data
        train_dataset = load_dataset("json", data_files=train_jsonl, split="train")
        eval_dataset = load_dataset("json", data_files=eval_jsonl, split="train")

        # LoRA training
        trainer = LoRATrainer(self.args, self.max_tokens)
        model, tokenizer = trainer.load_base_model_and_tokenizer()
        T1_train = time.perf_counter()
        model = trainer.train(model, tokenizer, train_dataset, eval_dataset, self.args.output_dir)
        T2_train = time.perf_counter()

        print("Training time: %s s" % (T2_train - T1_train))

        # Evaluate after training
        chat_engine = ChatEngine(model, tokenizer, max_tokens=self.max_tokens)
        tester = Tester(self.args, chat_engine)
        T1_test = time.perf_counter()
        num_pairs = tester.run(
            test_type="eval_after_ft",
            codebase_data=self.args.codebase_data_file,
            test_data=self.args.test_data_file,
            output_dir=self.args.output_dir,
        )
        T2_test = time.perf_counter()
        print("Inference time (per mutant pair): %s s" % ((T2_test - T1_test) / num_pairs))

    def _load_model_for_test(self):
        if self.args.test_type in ["zero-shot-prompt", "few-shot-prompt"]:
            # Use base model directly for zero/few-shot; 8-bit to save memory
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                load_in_8bit=True,
                device_map="auto",
                max_memory=self.max_memory,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name, trust_remote_code=True)
        else:
            # Inference from LoRA checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=self.max_memory,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, self.args.checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name)
        return model, tokenizer

    def run_test(self):
        model, tokenizer = self._load_model_for_test()
        chat_engine = ChatEngine(model, tokenizer, max_tokens=self.max_tokens)
        tester = Tester(self.args, chat_engine)

        T1 = time.perf_counter()
        num_pairs = tester.run(
            test_type=self.args.test_type,
            codebase_data=self.args.codebase_data_file,
            test_data=self.args.test_data_file,
            output_dir=self.args.output_dir,
        )
        T2 = time.perf_counter()
        print("Inference Time (per mutant pair): %s s" % ((T2 - T1) / num_pairs))
