import os

os.system("python -m QwenCoder.code.run \
    --output_dir=./QwenCoder/saved_models/Equivalence \
    --model_type=qwen2 \
    --model_name_or_path=./QwenCoder/models \
    --tokenizer_name=./QwenCoder/models \
    --do_test \
    --test_type=few-shot-prompt\
    --codebase_data_file=./dataset/Mutant_db_rip.csv \
    --train_data_file=./dataset/Mutant_A_rip.csv \
    --test_data_file=./dataset/Mutant_C_rip.csv \
    --seed 0")