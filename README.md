# Reproducing the invention of a named reaction: Zero-shot prediction of unseen chemical reactions
This is the code for "Reproducing the invention of a named reaction: Zero-shot prediction of unseen chemical reactions" paper.  The preprint of this paper can be found in ChemRxiv with https://doi.org/10.26434/chemrxiv.14034890.v1

# Requirments
## Python 2.7
## Tensorflow 1.11
## RDkit 2019.03.4

# Conda Environemt Setup
```
conda create -n transformer python=2.7
conda activate transformer
conda install -c rdkit rdkit=2019.03.4 -y
conda install -c tensorflow tensorflow=1.11.0 -y
git clone https://github.com/hongliangduan/Reproducing-the-invention-of-a-named-reaction-Zero-shot-prediction-of-unseen-chemical-reactions
```

# Dataset
The data for training, dev and testing of Zero-shot reaction prediction are provided in "Zero-shot reaction prediction data" file. 
The data for training, dev and testing of One-shot reaction prediction are provided in "One-shot reaction prediction data" file.
# Step 1: Preprocess the data
The input data can preprocessed by running the ``` sh datagen.sh ``` script, and the output data was in t2t_data folder.

# Train
Model training can be started by running the train.sh script.

# Test
Model testing can be started by running the decode.sh script.
