# 521 - Project - Exploring Different Word Embeddings for Next Word Prediction

## Directories & Files

The following is a description of the folders and files
included in the repository. Below is a graphical summary of the entire repo, 
following that is a more detailed explanation of the various files and or 
directories. Note that only data files on GitHub are the original dataset files
(`data/raw/ptbdataset/*.txt`) to conserve storage space. The rest of the data 
files are generated upon running the various python scripts in this repo.

📦`521-Project/`<br>
 ┣ 📂`data/`- Contains original (raw), intermediate and prediction data<br>
 ┃ ┣ 📂`ohe_vectors/` - OHE data set generated upon running `preprocess_text.py`<br>
 ┃ ┃ ┣ 📜`devX.npy` - One Hot Encoded predictor variables (development set)<br>
 ┃ ┃ ┣ 📜`devY.npy` - One Hot Encoded response variables (development set)<br>
 ┃ ┃ ┣ 📜`testX.npy` - One Hot Encoded predictor variables (test set)<br>
 ┃ ┃ ┣ 📜`testY.npy` - One Hot Encoded response variables (test set)<br>
 ┃ ┃ ┣ 📜`trainX.npy` - One Hot Encoded predictor variables (train set)<br>
 ┃ ┃ ┗ 📜`trainY.npy` - One Hot Encoded response variables (train set)<br>
 ┃ ┣ 📂`predicted/` - Model predictions, generated upon running `train_lstm_model*.py`<br>
 ┃ ┃ ┣ 📜`model1.npy` - LSTM model predictions that use word2vec embeddings<br>
 ┃ ┃ ┣ 📜`model2.npy` - LSTM model predictions that trains its own embeddings<br>
 ┃ ┃ ┗ 📜`model3.npy` - LSTM model predictions with no embedding space<br>
 ┃ ┣ 📂`processed/` - Contains the cleaned sentence tokens split into predictors and response sets<br>
 ┃ ┃ ┣ 📜`devX.csv` - Pre-processed predictor tokens (development set)<br>
 ┃ ┃ ┣ 📜`devY.csv` - Pre-processed response tokens (development set)<br>
 ┃ ┃ ┣ 📜`testX.csv` - Pre-processed predictor tokens (test set)<br>
 ┃ ┃ ┣ 📜`testY.csv` - Pre-processed response tokens (test set)<br>
 ┃ ┃ ┣ 📜`trainX.csv` - Pre-processed predictor tokens (train set)<br>
 ┃ ┃ ┗ 📜`trainY.csv` - Pre-processed response tokens (train set)<br>
 ┃ ┣ 📂`raw/` - The original un-processed data set<br>
 ┃ ┃ ┗ 📂`ptbdataset/` - The original un-processed Penn Treebank corpus<br>
 ┃ ┃ ┃ ┣ 📜`ptb.char.test.txt` - Character level test set (not used for project)<br>
 ┃ ┃ ┃ ┣ 📜`ptb.char.train.txt` - Character level train set (not used for project)<br>
 ┃ ┃ ┃ ┣ 📜`ptb.char.valid.txt` - Character level dev set (not used for project)<br>
 ┃ ┃ ┃ ┣ 📜`ptb.test.txt` - Word level raw test set<br>
 ┃ ┃ ┃ ┣ 📜`ptb.train.txt` - Word level raw train set<br>
 ┃ ┃ ┃ ┣ 📜`ptb.valid.txt` - Word level raw dev set<br>
 ┃ ┃ ┃ ┗ 📜`README` - Corpus information<br>
 ┃ ┣ 📂`vectors/` - Word2Vec representation of the Penn Treebank dataset<br>
 ┃ ┃ ┣ 📜`devX.npy` - Word2Vec predictor variables (development set)<br>
 ┃ ┃ ┣ 📜`devY.npy` - Word2Vec response variables (development set)<br>
 ┃ ┃ ┣ 📜`testX.npy` - Word2Vec predictor variables (test set)<br>
 ┃ ┃ ┣ 📜`testY.npy` - Word2Vec response variables (test set)<br>
 ┃ ┃ ┣ 📜`trainX.npy` - Word2Vec predictor variables (train set)<br>
 ┃ ┃ ┗ 📜`trainY.npy` - Word2Vec response variables (train set)<br>
 ┃ ┗ 📜`tok_arr.npy` - Represents the keys of the tokens to word index lookup dictionary<br>
 ┣ 📂`models/` - The trained LSTM model TF files and Word2Vec model<br>
 ┃ ┣ 📜`LSTM1.h5` - LSTM model that use word2vec embeddings<br>
 ┃ ┣ 📜`LSTM2.h5` - LSTM model that trains its own embeddings<br>
 ┃ ┣ 📜`LSTM3.h5` - LSTM model with no embedding space<br>
 ┃ ┗ 📜`word2vec.model` - Trained Word2Vec model trained on corpus<br>
 ┣ 📂`plots/` - Model training results plots<br>
 ┃ ┣ 📜`acc_results.png` - Summary accuracy plot of all 3 models<br>
 ┃ ┣ 📜`loss1.png` - LSTM model 1 training results<br>
 ┃ ┣ 📜`loss2.png` - LSTM model 2 training results<br>
 ┃ ┗ 📜`loss3.png` - LSTM model 3 training results<br>
 ┣ 📜`evaluate.py` - Evaluates various metrics of model predictions<br>
 ┣ 📜`preprocess_text.py` - Preprocessed the original raw data set<br>
 ┣ 📜`README.md` - This file<br>
 ┣ 📜`requirements.txt.` - Required modules to run repo scripts<br>
 ┣ 📜`train_lstm_model1.py` - Source script for LSTM model 1<br>
 ┣ 📜`train_lstm_model2.py` - Source script for LSTM model 2<br>
 ┣ 📜`train_lstm_model3.py` - Source script for LSTM model 3<br>
 ┗ 📜`util.py` - Various helper functions to process datasets in `preprocess_text.py`<br>


## Data

The "data" directory is responsible for containing the
subdirectories related to the various stages of data generated
by carrying out the project. Below is a brief description of
each data subdirectory.

* <b>Raw:</b> raw textual data made available by the Penn Treebank Corpus
* <b>Processed:</b> intermediate textual data that has been preprocessed and awaits to be transformed
* <b>OHE_Vectors:</b> one-hot-encoded data used as inputs for training the neural network models
* <b>Predicted:</b> vectorized data representing predictions made by the neural network models

Note that only the "raw" data subdirectory is included in the
repository, for the other data subdirectories are generated by
running the Python scripts necessary to reproduce the project.
<br><br>


### Raw Data Files

The following is a brief description of the Penn Treebank data
folder and associated files included in the "raw" data
subdirectory of the repository.

* <b>ptbdataset:</b> the folder containing the files of the Penn Treebank Corpus
* `ptb.char.test.txt`: the character-level test set (not used for the purposes of this project)
* `ptb.char.train.txt`: the character-level train set (not used for the purposes of this project)
* `ptb.char.valid.txt`: the character-level development set (not used for the purposes of this project)
* `ptb.test.txt`: the word-level test set used to generate vectors for evaluating each model
* `ptb.train.txt`: the word-level train set used to generate vectors for training each model
* `ptb.valid.txt`: the word-level development set used to generate vectors for adjusting the hyperparameters of each model
* `README`: the original README file used to describe the Penn Treebank Corpus
<br><br>


## Models


## Environment Setup

Clone the repo and navigate to the `521-Project` directory. If using Anaconda or 
Miniconda create a new environment with `python` version 3.9.7.

`conda create --name nextwordpred python=3.9.12`

Activate the environment

`conda activate nextwordpred`

`tensorflow` can be a little tricky to get up and running sometimes. It is 
recommended that all other modules first be installed with `conda` then install
`tensorflow` with `pip`

`conda install keras gensim matplotlib pandas`

`pip install tensorflow`

**Note:** if having trouble installing `matplotlib` attempt to install from the 
`conda-forge` channel

`conda install -c conda-forge matplotlib`

## Usage Examples

Note that `preprocess_text.py` should be run before training or evaluating LSTM models.


`python preprocess_text.py /`<br>
&emsp;&emsp;`--ptb_train_file data/raw/ptbdataset/ptb.train.txt /`<br>
&emsp;&emsp;`--ptb_dev_file data/raw/ptbdataset/ptb.valid.txt /`<br>
&emsp;&emsp;`--ptb_test_file data/raw/ptbdataset/ptb.test.txt`<br>

`python train_lstm_model1.py /`<br>
&emsp;&emsp;`--vector_directory data/vectors /`<br>
&emsp;&emsp;`--ohe_directory data/ohe_vectors /`<br>
&emsp;&emsp;`--epochs 100 /`<br>
&emsp;&emsp;`--units 100 /`<br>
&emsp;&emsp;`--activation tanh /`<br>
&emsp;&emsp;`--recurrent_activation sigmoid /`<br>
&emsp;&emsp;`--dense_activation relu /`<br>
&emsp;&emsp;`--input_shape (5, 100) /`<br>
&emsp;&emsp;`--model_file models/LSTM1.h5`<br>

`python train_lstm_model2.py /`<br>
&emsp;&emsp;`--ohe_directory data/ohe_vectors /`<br>
&emsp;&emsp;`--seq_directory data/processed /`<br>
&emsp;&emsp;`--epochs 30 /`<br>
&emsp;&emsp;`--units 100 /`<br>
&emsp;&emsp;`--activation tanh /`<br>
&emsp;&emsp;`--recurrent_activation sigmoid /`<br>
&emsp;&emsp;`--dense_activation relu /`<br>
&emsp;&emsp;`--model_file models/LSTM2.h5`<br>

`python train_lstm_model3.py /`<br>
&emsp;&emsp;`--vector_directory data/ohe_vectors /`<br>
&emsp;&emsp;`--epochs 20 /`<br>
&emsp;&emsp;`--units 100 /`<br>
&emsp;&emsp;`--activation tanh /`<br>
&emsp;&emsp;`--recurrent_activation sigmoid /`<br>
&emsp;&emsp;`--dense_activation relu /`<br>
&emsp;&emsp;`--input_shape (5, 9999) /`<br>
&emsp;&emsp;`--model_file models/LSTM3.h5`<br>

`python evaluate.py /`<br>
&emsp;&emsp;`--w2v_model_file models/word2vec.model /`<br>
&emsp;&emsp;`--ohe_vectors data/ohe_vectors /`<br>
&emsp;&emsp;`--predicted_vectors data/predicted`<br>
&emsp;&emsp;`--token_arr data/tok_arr.npy`<br>