import os


os.system("python -m CodeBERT.code.run \
        --output_dir=./CodeBERT/saved_models/Equivalence \
        --config_name=./CodeBERT/codebert-base \
        --model_name_or_path=./CodeBERT/codebert-base \
        --tokenizer_name=./CodeBERT/codebert-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=./dataset/Mutant_db_rip.csv \
        --train_data_file=./dataset/Mutant_A_rip.csv \
        --eval_data_file=./dataset/Mutant_C_rip.csv \
        --test_data_file=./dataset/Mutant_C_rip.csv 2>&1")