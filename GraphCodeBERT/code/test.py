import os


os.system("python -m GraphCodeBERT.code.run \
        --output_dir=./GraphCodeBERT/saved_models/Equivalence \
        --config_name=./GraphCodeBERT/graphcodebert-base \
        --model_name_or_path=./GraphCodeBERT/graphcodebert-base \
        --tokenizer_name=./GraphCodeBERT/graphcodebert-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=./dataset/Mutant_db_rip.csv \
        --train_data_file=./dataset/Mutant_A_rip.csv \
        --eval_data_file=./dataset/Mutant_C_rip.csv \
        --test_data_file=./dataset/Mutant_C_rip.csv 2>&1")