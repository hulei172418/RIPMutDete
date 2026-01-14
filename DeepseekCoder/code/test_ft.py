import os

os.system("python -m DeepseekCoder.code.run \
    --output_dir=./DeepseekCoder/saved_models/Equivalence \
    --model_type=llama \
    --model_name_or_path=./DeepseekCoder/models \
    --tokenizer_name=./DeepseekCoder/models \
    --do_test \
    --checkpoint_dir=./DeepseekCoder/saved_models/checkpoints/\
    --codebase_data_file=./dataset/Mutant_db_rip.csv \
    --train_data_file=./dataset/Mutant_A_rip.csv \
    --test_data_file=./dataset/Mutant_C_rip.csv \
    --seed 0")