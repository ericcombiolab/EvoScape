# EvoScape
This repository contains the source code, model weights of EvoScape, as well as the 11 collected deep mutational scanning (DMS) datasets. EvoScape is a computational model that predicts the immune escape potential of viral variants relative to a wild-type sequence. Its applications include identifying immune escape hotspots and forecasting future circulating viral strains, making it a valuable tool for surveillance during epidemics and pandemics.

# Usage
## Install EvoScape
```
git clone https://github.com/ericcombiolab/EvoScape.git
conda env create -f evoscape_environment.yml
```
## Train EvoScape
The script for training EvoScape is in ```DDP_training.py```. Before training EvoScape, you need to set the user-defined inputs first. They are listed and explained as follows:
```
ratio: the ratio of positive samples to negative samples in the training set
os.environ['CUDA_VISIBLE_DEVICES']: this sets the GPUs to be used for training. For example, '0,1' means we use GPU 0 and GPU 1 for training, while '0' means we only use GPU 0 for training.
data_path: the path to the training dataset
esmfold_path: the path to the finetuned ESMFold model
saved_model_prefix: the prefix for the saved model checkpoints, the saved model will be named as {model_prefix}_epoch{epoch}.pt
batch_size: the batch size for training
training_epoch: the number of epochs for training
```
After setting these inputs, you can train EvoScape by running:
```
python DDP_training.py
```
## EvoScape inference
The script for using EvoScape for inference is in ```inference.py```. Before running inference, you need to set the user-defined inputs first. They are listed and explained as follows:
```
batch_size: the batch size for training
device_id: the GPU ID for inference. For example, it can be set to 0 if you would llike to use GPU 0 for inference.
data_path is the path to the test dataset
esmfold_path is the path to the finetuned ESMFold model
state_dict_path is the path to the saved model during the training stage
output_path is the path to the output file
```
After setting these inputs, you can run inference by the following:
```
python inference.py
```
## File formats required by EvoScape
For training, EvoScape requires three types of inputs: wild-type sequences, mutant sequences, and labels. They should be stored in a csv file as follows:
```
raw_seq,mut_seq,label
...,...,...
```
For inference, EvoScape requires two types of inputs: wild-type sequences, mutant sequences. They should be stored in a csv file as follows:
```
raw_seq,mut_seq
...,...
```
We provide sample training and test datasets for EvoScape in ```data/evoscape_train_test/```. Note that there are additional columns in the sample data, but they are not a must for EvoScape.
# DMS datasets
In total, 11 DMS datasets were collected. You may find their sources in the supplementary file of the EvoScape paper. The raw files of these datasets are stored in ```data/raw_DMS```. The processed files are stored in ```data/processed_DMS```, which have a uniform format as follows:
```
pos,wildtype,mutant,label
...,...,...,...
```
# Citation
If you would like to cite EvoScape, please cite it as follows:
