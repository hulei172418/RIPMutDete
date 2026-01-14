import os
import sys


print(">>> test.py sys.executable =", sys.executable)
os.system("python -m PLBART.code.run \
        --output_dir=./PLBART/saved_models/Equivalence \
        --config_name=./PLBART/plbart-base \
        --model_name_or_path=./PLBART/plbart-base \
        --tokenizer_name=./PLBART/plbart-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=./dataset/Mutant_db_rip.csv \
        --train_data_file=./dataset/Mutant_A_rip.csv \
        --eval_data_file=./dataset/Mutant_C_rip.csv \
        --test_data_file=./dataset/Mutant_C_rip.csv 2>&1")