# Helios: Learning and Adaptation of Matching Rules for Continual In-Network Malicious Traffic Detection
![overview](./figs/overview.png)

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

## Training:
To train the prototype network, simply run the train.py script.
```python
python train.py
```
Models are saved in ./checkpoints.

## Boosting & Incremental Process:
The `incremental_boost.py` script handles the training process during class incremental learning.
```python
python incremental_boost.py
```
If you need to change the class incremental order or modify the known classes, adjust the `--selected_class` and `--add_order` parameters.

The boosting and incremental training process typically takes around 180 seconds.
