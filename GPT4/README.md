# Replication of GPT-4 on Equivalent Mutant Detection

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,Mutant_db_rip.csv), and 3 mutant-pair files for training/validation/testing (i.e.,Mutant_A_rip.csv, Mutant_B_rip and Mutant_C_rip.csv) in the `../dataset` folder.

### Testing:

**Inference of Zero-shot Prompting**
Before starting inference, please add your OpenAI API key to the `api_key` and `base_url` in `./code/test_zsp.py`.  
To run GPT-4 with Zero-shot Prompting to make inferences on the test dataset, run the following commands:

```
cd EquivalentMutantsLLM\EquivDetect
python -m GPT4.code.test_zsp
```

**Inference of Few-shot Prompting**
Before starting inference, please add your OpenAI API key to the `api_key` and `base_url` in `./code/test_fsp.py`.  
To run GPT-4 with Few-shot Prompting to make inferences on the test dataset, run the following commends:

```
cd EquivalentMutantsLLM\EquivDetect
python -m GPT4.code.test_fsp
```
