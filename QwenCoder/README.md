# Replication of QwenCoder on Equivalent Mutant Detection

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,Mutant_db_rip.csv), and 3 mutant-pair files for training/validation/testing (i.e.,Mutant_A_rip.csv, Mutant_B_rip and Mutant_C_rip.csv) in the `../dataset` folder.

### (1) Training

You can train the QwenCoder through the following commands:

```
cd EquivalentMutantsLLM\EquivDetect
python -m QwenCoder.code.train
```

Then, the program will automatically train the QwenCoder via SFT manner and do inference after training.

### (2) Testing:

**Inference of Zero-shot Prompting**  
To run QwenCoder with Zero-shot Prompting to make inferences on the test dataset, run the following commands:

```
cd EquivalentMutantsLLM\EquivDetect
python -m QwenCoder.code.test_zsp
```

**Inference of Few-shot Prompting**
To run QwenCoder with Few-shot Prompting to make inferences on the test dataset, run the following commends:

```
cd EquivalentMutantsLLM\EquivDetect
python -m QwenCoder.code.test_fsp
```

**Inference of Fine-tuning with Instruction**  
Before performing inference with the instruction-tuned model, please return to `Step (1)` and run the train.py script to complete the LoRA fine-tuning of QwenCoder. The fine-tuned model parameters will be automatically saved in the `save_models/` directory.
After completing the training process, you can run the following command to perform inference:

```
cd EquivalentMutantsLLM\EquivDetect
python -m QwenCoder.code.test_ft
```
