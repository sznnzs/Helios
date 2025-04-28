# Helios: Learning and Adaptation of Matching Rules for Continual In-Network Malicious Traffic Detection
![overview](./overview.png)

First, run:
```sh
tar -zvxf ./dataset.tar.gz
```

## Project Structure:
```bash
|-- checkpoints/                        # Directory to save trained .pth files
|-- dataset/                            # Directory for the dataset
|   |-- CSE-CICIDS-2018-improved/       # Original dataset files
|   |-- preprocess_CICIDS_2018.py       # CICIDS preprocessing script
|   |-- CICIDS_2018_X.npy               # Processed feature data
|   |-- CICIDS_2018_y.npy               # Processed label data
|-- utils/                              # Directory containing utility functions
|-- train.py                            # Script to train the prototype network
|-- incremental_boost.py                # Script for complete training of the prototype network, boosting, and the incremental process
```

## Dataset:


You can download the original dataset from the following link:
https://www.unb.ca/cic/datasets/ids-2018.html or https://github.com/Ruoyu-Li/UAD-Rule-Extraction/tree/main/dataset/CSE-CICIDS-2018-improved, then use preprocess_CICIDS_2018.py to preprocess.

If you need to use a custom dataset, modify the ./utils/get_data.py script.

## Training the Prototype Network

To perform an initial training of the prototype network on the known classes, execute the `train.py` script:

```
python train.py [arguments]
```

Key arguments for `train.py` (and also relevant for `incremental_boost.py`) include:

- `--gpu`: Index of the GPU to use.
- `--dataset`: Name of the dataset to use.
- `--selected_class`: List of initial known class indices.
- `--epochs`: Number of training epochs.
- `--batch_size`: Training batch size.
- `--learning_rate`: Optimizer learning rate.
- `--init_type`: Prototype initialization method.
- `--save_model`: Path prefix for saving model checkpoints.

Trained models from this script will be saved in the `./checkpoints/` directory. This step primarily serves to establish an initial model baseline.

## Boosting and Incremental Learning

The core logic, including prototype network training, boosting iterations, rule generation, pruning, and incremental updates for new classes, is handled by the `incremental_boost.py` script.

Run the script using:

```
python incremental_boost.py [arguments]
```

This script iteratively performs the following steps:

1. **Initializes/Loads Model:** Starts with an initial model or loads a previously saved one.
2. **Boosting Iterations (`--boost_num`):**
   - Trains the prototype network on the current set of training samples (initially all known data, later residual samples).
   - Prunes insignificant prototypes based on support (`--prune_T`).
   - Calculates distance thresholds (`--threshold_radio`) and converts prototypes to range rules.
   - Merges newly generated rules with existing rules.
   - Identifies residual samples (misclassified or unclassified by current rules) to be used in the next boosting iteration.
3. **Rule Pruning:** Optionally prunes rules with low sample coverage (`--prune_rule`).
4. **Evaluation:** Tests the generated ruleset on the test data and records performance metrics (Accuracy, Precision, Recall, F1, TPR, FPR) in `results.csv`.
5. **Incremental Update (`--add_order`):**
   - If new classes are specified in `--add_order`, it identifies and removes rules from the existing set that conflict with samples from the new class.
   - Prepares the training data for the next incremental step, potentially including samples from the newly introduced class and residual samples from the previous stage.
   - Updates the `--selected_class` list.

**Key Arguments for Incremental Learning:**

- `--boost_num`: Number of boosting iterations within each incremental step.
- `--add_order`: A list defining the order in which new class indices are introduced incrementally. Use `-1` to signify the end or no new classes in a step.
- `--prune_T`: Threshold for pruning prototypes based on sample support.
- `--prune_rule`: Threshold for pruning rules based on sample coverage.
- `--threshold
