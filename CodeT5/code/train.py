import os


os.system("python -m CodeT5.code.run \
        --output_dir=./CodeT5/saved_models/Equivalence \
        --config_name=./CodeT5/codet5-base \
        --model_name_or_path=./CodeT5/codet5-base \
        --tokenizer_name=./CodeT5/codet5-base \
        --requires_grad 0 \
        --do_train \
        --code_db_file=./dataset/Mutant_db_rip.csv \
        --train_data_file=./dataset/Mutant_A_rip.csv \
        --eval_data_file=./dataset/Mutant_B_rip.csv \
        --test_data_file=./dataset/Mutant_B_rip.csv 2>&1")