# Replication of CodeT5 on Equivalent Mutant Detection

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,Mutant_db_rip.csv), and 3 mutant-pair files for training/validation/testing (i.e.,Mutant_A_rip.csv, Mutant_B_rip and Mutant_C_rip.csv) in the `../dataset` folder.

### (1) Training

You can train the original model through the following commands:

```
cd EquivalentMutantsLLM\EquivDetect
python -m CodeT5.code.train
```

### (2) Testing

To run our fine-tuned model to make inferences on the test dataset, run the following commands:

```
cd EquivalentMutantsLLM\EquivDetect
python -m CodeT5.code.test
```

_Note 1:_ Before performing inference with the instruction-tuned model, please go back to **Step (1)** and run the training script to fine-tune CodeT5. After completing the training process, you can perform inference by following the procedure in **Step (2)**.

_Note 2:_ In train.py and test.py, `--requires_grad=0` indicates pre-trained code embedding strategy, and `--requires_grad=1` indicates fine-tuned code embedding strategy.
