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
The data for training, dev and testing of Zero-shot reaction prediction are provided in ```Zero-shot reaction prediction data``` file. 
The data for training, dev and testing of One-shot reaction prediction are provided in ```One-shot reaction prediction data``` file.

# Step 1: Preprocess the data
Make ```tmp/t2t_datagen``` folder, and put data in this folder.
The input data can preprocessed by running the ``` sh datagen.sh ``` script, make ```t2t_data```, ```t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu```folders
## After running the preprocessing, the following files are generated:
```vocab.source```:the vocab of reactants.
```vocab.target```:the vocab of products.
and other outputs was in ```t2t_data``` fold.

# Step 2: Train the model
## After running the training, the following files are generated:
```model.ckpt```: moldel generated ckpt was put in ```t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu```folders 
## Zero-short
### Train
Combining the data in ```Zero-shot reaction prediction data/Data for training/Training set_USPTO dataset/``` floder and the data ```Zero-shot reaction prediction data/Data for training/Fine-tuning set for Barton and Suzuki reactions/``` folder.
Run the ```sh train.sh``` script.
Run Step 3.
## One-short
### Pretrain
Combining the data in ```One-shot reaction prediction data/Data for training/Training set_USPTO dataset/``` folder and the data ```One-shot reaction prediction data/Data for training/Fine-tuning set for Barton and Suzuki reactions/``` folder.
Run the ```sh train.sh``` script.
### Transfer_learning
Using the data in ```One-shot reaction prediction data/Data for fine-tuning/``` folder and the ```model.ckpt``` films in ```t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu```folders generated by One-short pretrain.
Run the ```sh train.sh``` script.
Run Step 3.


# Step 3: Chemical reaction prediction
## Zero-short 
Using the data in ```Zero-shot reaction prediction data/Data for test/decode_this.txt``` film and the ```model.ckpt```film in  ```t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu```folders generated by Zero-short train.
Run the ```sh decode.sh``` script.

## One-short
Using the data in ```One-shot reaction prediction data/Data for test/decode_this.txt``` film and the ```model.ckpt```film in  ```t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu```folders generated by One-short transfer-learning.
Run the ```sh decode.sh``` script.
