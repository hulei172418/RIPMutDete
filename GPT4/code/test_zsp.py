import os


api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # The API key provided by the model or proxy platform
base_url = "https://api.openai.com/v1"  # The URL provided by the service provider

os.system(f"python -m GPT4.code.run \
    --api_key={api_key}\
    --base_url={base_url}\
    --output_dir=./GPT4/saved_models/Equivalence \
    --model_type=gpt-4.1-mini \
    --model_name_or_path=./GPT4/models \
    --tokenizer_name=./GPT4/models \
    --do_test \
    --test_type=zero-shot-prompt\
    --codebase_data_file=./dataset/Mutant_db_rip.csv \
    --train_data_file=./dataset/Mutant_A_rip.csv \
    --test_data_file=./dataset/Mutant_C_rip.csv \
    --seed 0")