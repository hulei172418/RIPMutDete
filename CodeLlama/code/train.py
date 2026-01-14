import os

os.system("python -m CodeLlama.code.run \
    --output_dir=./CodeLlama/saved_models/Equivalence \
    --model_type=llama \
    --model_name_or_path=./CodeLlama/models \
    --tokenizer_name=./CodeLlama/models \
    --do_train \
    --codebase_data_file=./dataset/Mutant_db_rip.csv \
    --train_data_file=./dataset/Mutant_A_rip.csv \
    --test_data_file=./dataset/Mutant_B_rip.csv \
    --epoch 10 \
    --gradient_accumulation_steps 4\
    --train_batch_size 2 \
    --learning_rate 2e-5 \
    --seed 0")