import os
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


class LoRATrainer:
    """
    Stable fine-tuning wrapper for DeepSeek/LLama series + LoRA
    - 4bit quantization + kbit training preparation to avoid known compatibility issues with int8
    - Multi-GPU device_map=auto + per-card max_memory to fully utilize n × GPU
    - Uses formatting_func to process {"messages":[...]} format data
    """

    def __init__(self, args, max_tokens):
        self.args = args
        self.max_tokens = max_tokens

    def load_base_model_and_tokenizer(self):
        # 4bit quantization config
        bnb4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Reserve ~2GiB per GPU for system
        n_gpus = torch.cuda.device_count()
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory/ (1024 ** 3)*0.9:.2f}GiB" \
            for i in range(n_gpus)} if n_gpus > 0 else None

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            quantization_config=bnb4,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # Disable use_cache for training with gradient_checkpointing
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.tokenizer_name, use_fast=True, trust_remote_code=True
        )
        # Align pad_token and use right padding for better long sequence training
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def train(self, model, tokenizer, train_dataset, eval_dataset, output_dir: str):
        # Prepare for 4bit/kbit training (instead of prepare_model_for_int8_training)
        model = prepare_model_for_kbit_training(model)

        # LoRA config: override common linear layers in LLaMA/DeepSeek
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        ckpt_dir = os.path.join(output_dir, "../checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        training_args = TrainingArguments(
            per_device_train_batch_size=self.args.train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            bf16=use_bf16,
            fp16=not use_bf16,
            eval_strategy="epoch",
            logging_strategy="epoch",
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=ckpt_dir,
            load_best_model_at_end=False,
            gradient_checkpointing=True,
            report_to="none",
            run_name=f"{self.args.model_type}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            save_safetensors=False,
            lr_scheduler_type="cosine",
        )

        def formatting_func(examples):
            rendered_texts = []
            msgs_batch = examples.get("messages", [])

            for msgs in msgs_batch:
                # Robustness: accept raw string if already plain text
                if isinstance(msgs, str):
                    rendered_texts.append(msgs)
                    continue

                # Normal path: messages is list[dict], render via chat_template
                try:
                    token_count = len(tokenizer.encode(msgs))
                    txt = tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,  # SFT usually excludes assistant prompt head
                    )
                except Exception:
                    # Fallback: no chat_template or malformed format, concatenate manually
                    parts = []
                    if isinstance(msgs, list):
                        for m in msgs:
                            if isinstance(m, dict):
                                role = m.get("role", "user")
                                content = m.get("content", "")
                                parts.append(f"<|{role}|>\n{content}")
                            else:
                                parts.append(str(m))
                    else:
                        parts.append(str(msgs))
                    txt = "\n".join(parts)
                rendered_texts.append(txt)

            return rendered_texts

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,  # Use formatting function instead of dataset_text_field
            max_seq_length=getattr(self.args, "max_seq_length", self.max_tokens),
            packing=False,
        )

        trainer.train()

        # Save only LoRA adapter
        adapter_dir = os.path.join(ckpt_dir, "adapter_model")
        os.makedirs(adapter_dir, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)
        trainer.model.eval()
        return trainer.model
