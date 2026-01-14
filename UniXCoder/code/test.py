import os


os.system("python -m UniXCoder.code.run \
        --output_dir=./UniXCoder/saved_models/Equivalence \
        --config_name=./UniXCoder/unixcoder-base \
        --model_name_or_path=./UniXCoder/unixcoder-base \
        --tokenizer_name=./UniXCoder/unixcoder-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=./dataset/Mutant_db_rip.csv \
        --train_data_file=./dataset/Mutant_A_rip.csv \
        --eval_data_file=./dataset/Mutant_C_rip.csv \
        --test_data_file=./dataset/Mutant_C_rip.csv 2>&1")