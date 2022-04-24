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
* <b>OHE_Vectors:</b> one-hot-encoded (OHE) data used as input (depending on the model) and output for training, validating, and testing the neural network models
* <b>W2V_Vectors:</b> Word2Vec vectors used as input (depending on the model) for training, validating, and testing the neural network models
* <b>Predicted:</b> vectorized data representing predictions made by the neural network models
<br>

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

The "models" directory houses all the model files created in
the project. This consists of a trained Gensim Word2Vec model
used to create word embeddings and three trained bidirectional
Long Short-Term Memory (LSTM) neural network models used to
make next-word predictions. Further information on these
models is given below.
<br><br>


### Word2Vec Model

In the context of natural language processing (NLP),
"Word2Vec," typically abbreivated as "W2V," generally refers
to a method for converting each word in a corpus of text to a
vector. In most cases, this is implemented using a neural
network model architecture that inputs tokens and outputs
vectors for the tokens based on their co-occurrence in the
corpus. In this project, the `word2vec.model` file contains a
model very similar to the one described above. However, rather
than developing an original implementation, the model in this
repository was made using a topic modeling and NLP library,
called "Gensim."
<br><br>

#### Objective & Modeling Process

To create the word embedding space, the W2V model was trained
using the skip-gram approach, in which the goal is to predict
a set of target context tokens based on a given input token.
The result of using this method was the generation of a unique
100-dimensional numerical vector for each token. These token
vectors were then used to transform the textual data into a
vectorized representation that the neural network models could
use for training and making predictions.
<br><br>


### Neural Network Models

As indicated previously, a total of three separate neural
network models were used in this project. Specifically, these
were recurrent neural networks (RNNs). RNNs are known for
their ability to handle sequential data, and this attribute
makes them very useful when it comes to next-word prediction
and other language modeling tasks. However, since the training
signal used to update the model weights is time-dependent in
the case of an RNN, longer input sequences cause this signal
to become drastically reduced during backpropagation when the
model learns. This effect is known as the vanishing gradient
problem, and it stymies RNNs in instances where longer
sequences are used as inputs. Thankfully, this problem can be
overcome by using a special instance of an RNN, known as an 
LSTM. This type of architecture has the capacity to preserve
long-term information about the sequential data it is exposed
to. Therefore, in this project, LSTM networks, as opposed to
the more broadly-defined RNN implementation, were used
instead. Lastly, all three models incorporated bidirectional
layers to allow for each of them to observe the sequences in
both the original order and the reversed order.
<br><br>


#### Objective & Modeling Process

In general, the objective of each model was to ingest a set of
input words and successfully predict the word that would be
most likely to follow. To execute this task, each model was
trained using a vectorized representation of five-token
sequences as input, along with the corresponding OHE target
token vector as output, using tokens that derived from
`ptb.train.txt`. For a training period of 100 epochs, loss,
in the form of categorical cross-entropy, and accuracy, in the
form of mean squared error, was monitored for both the
training set and `ptb.valid.txt` (the development set). The
hyperparameters of the models were tuned on tokens from
`ptb.valid.txt`, using the same input-output representation as
the training data. Through a combination of additional layers
with varying numbers of units and types of activation
functions, each model eventually arrived at a dense layer that
utilized a softmax activation function to predict the next
word by outputting a vector of probability scores across the
entire target vocabulary set. In this manner, the predicted
next word was the token in the target vocabulary set that was
assigned the highest probability by the model.
<br>

Overall, the training and prediction process described above
was non-unique to the models used in the project. Instead, the
representation of the data prior to input into bidirectional
layers was the primary factor that distinguished the models.
More details on this are included in the brief descriptions of
each model that are given below.
<br><br>


#### LSTM Model #1

The first of the three models created in the project is
represented by the `LSTM1.h5` model file. This model stood out
from the others in that it used the token vectors created from
the W2V model as input. In other words, each input sample fed
to the model consisted of a sequence of 5 100-dimensional W2V
vectors that represented the 5 tokens of the corresponding
preprocessed sentence.
<br><br>


#### LSTM Model #2

The second of the three models created in the project is
represented by the `LSTM2.h5` model file. Unlike the first
model, the second model did not make use of the pre-trained
word embeddings made available by the W2V model. Instead, each
input to the second model was a sequence of 5
10,000-dimensional OHE vectors. Using this representation, the
size of each vector was equivalent to the size of the
vocabulary of the entire Penn Treebank data set. Then, for
enhancement, the model was configured to train its own
embeddings by including an embedding layer in its
architecture. This layer acted to convert the sparse, OHE
vectors into dense, 100-dimensional vectors before passing
them along to the bidirectional LSTM layers.
<br><br>


#### LSTM Model #3

The third and final model created in the project is
represented by the `LSTM3.h5` model file. Similar to the
second model, each input for the third model consisted of 
sequences of 5 10,000-dimensional OHE vectors. However,
dissimilar to the second model, an embedding layer was not
used. Instead, these OHE vectors were fed directly into the
bidirectional LSTM layers of the neural network. Therefore,
the final model produced in the project served as a baseline
to use for judging the performance of the previous models.
<br><br>


## Plots

The "plots" directory holds all the visualizations that were
generated by carrying out the project. For each model, line
plots were included to help demonstrate how the loss on the
training and validation sets changed as a function of the
number of epochs. In addition, a bar chart was made to
showcase the accuracies of each model in terms of two
quantities: the count of times the predicted next word was the
actual next word, and the count of times the predicted next
word was in the top 10 most similar words of the actual word
(including the actual word). Below is a brief description of
each plot file.

* `acc_results.png`: a bar chart summarizing model accuracies
* `loss1.png`: a line plot of the loss on the training and validation sets for the first model
* `loss2.png`: a line plot of the loss on the training and validation sets for the second model
* `loss3.png`: a line plot of the loss on the training and validation sets for the third model
<br><br>


## Script Files

The following is a description of the Python script files that
were used to form the data pipeline necessary to carry out the
project.
<br><br>


### `util.py`

This file contains many utility functions for executing
various data operations. This includes data loading,
preprocessing, splitting, writing, and other related tasks.
Below, documentation of each function in `util.py` is given.

## Functions:

### `load_sents(raw_data_path)`

### <b>Description:</b>

This function converts a newline-separated text file of
sentences to a nested list.

### <b>Arguments:</b>

| Name            | Type  | Description                            |
|-----------------|-------|----------------------------------------|
 | `raw_data_path` | `str` | The path to the text file of sentences | 

### <b>Returns:</b>

| Name    | Type                        | Description                              |
|---------|-----------------------------|------------------------------------------|
 | `sents` | `list[list[str, str, ...]]` | A list of lists of sentence-level tokens |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `preprocess_sents(sents, fixed_length=False, length=2)`

### <b>Description:</b>

This function preprocesses a list of sentences. It also
includes the option of forcing all sentences to contain the
same number of tokens.

### <b>Arguments:</b>

| Name           | Type                        | Description                                            |
|----------------|-----------------------------|--------------------------------------------------------|
 | `sents`        | `list[list[str, str, ...]]` | A list of lists of sentence-level tokens               |
 | `fixed_length` | `bool`                      | A flag denoting whether to fix sentence length         |
 | `length`       | `int`                       | The number of tokens that each sentence should include |

### <b>Returns:</b>

| Name            | Type                        | Description                                             |
|-----------------|-----------------------------|---------------------------------------------------------|
 | `sents_preproc` | `list[list[str, str, ...]]` | A list of lists of preprocessed sentence-level tokens   |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `split_sentences(sentences, length)`

### <b>Description:</b>

This function splits sentences into predictor and response
variables depending on the desired length of sentences. The
response variable is always the last token to be kept,
indicated by the passed sentence length argument.

### <b>Arguments:</b>

| Name        | Type                        | Description                                                                                                                                              |
|-------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
 | `sentences` | `list[list[str, str, ...]]` | A 2-dimensional iterable in which the first dimension is each sentence and the 2nd dimension is a list of tokens in the sentence, represented as strings |
 | `length`    | `int`                       | The total number of tokens to keep in the sentence/phrase. This value should be the total of the predictor tokens and the response token                 |

### <b>Returns:</b>

| Name   | Type    | Description                                                                                                                                                                                              |
|--------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | `X, Y` | `tuple` | The predictor and response tokens represented as 2-dimensional iterables in which the first dimension is each sentence and the 2nd dimension is a list of tokens in the sentence, represented as strings |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `write_processed(sentences, fname)`

### <b>Description:</b>

This function saves the preprocessed sentences in token format
to csv files after converting them to a data frame.

### <b>Arguments:</b>

| Name        | Type                        | Description                                                                                                          |
|-------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------|
 | `sentences` | `list[list[str, str, ...]]` | A list of lists where the first dimension is a list of sentences and the second is a list of tokens in string format |
 | `fname`     | `str`                       | The system file path to save the processed tokens                                                                    |

### <b>Returns:</b>

This function does not return anything.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `word_2_vector(sentences, keyed_vectors, file, v_type)`

### <b>Description:</b>

This function converts lists of sentences from word tokens to
word vectors or response words to a vector. It also possesses
the functionality to save the vectors to an output file.

### <b>Arguments:</b>

| Name            | Type                                      | Description                                                                                                          |
|-----------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
 | `sentences`     | `list[list[str, str, ...]]`               | A 2-dimensional array where the first dimension is each sentence and the 2nd dimension is each token in the sentence |
 | `keyed_vectors` | `gensim.models.keyedvectors.KeyedVectors` | The word2vec model which should be used to convert tokens to vectors                                                 |
 | `file`          | `str`                                     | The relative system file path of where to save the sentence vectors                                                  |
 | `v_type`        | `str`                                     | The type of variable (either "predictor" or "response")                                                              |

### <b>Returns:</b>

This function does not return anything.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `form_dictionary(sents)`

### <b>Description:</b>

This function uses sentences to form a dictionary mapping
sentence-level tokens to unique indexes.

### <b>Arguments:</b>

| Name     | Type                        | Description                              |
|----------|-----------------------------|------------------------------------------|
 | `sents`  | `list[list[str, str, ...]]` | A list of lists of sentence-level tokens |

### <b>Returns:</b>

| Name       | Type             | Description                            |
|------------|------------------|----------------------------------------|
 | `tok_dict` | `dict{str: int}` | A dictionary mapping tokens to indexes |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `transform_sents(sents, tok_dict)`

### <b>Description:</b>

This function uses a token dictionary to form a
dictionary-index-mapped representation of the tokens in each
sentence

### <b>Arguments:</b>

| Name        | Type                        | Description                              |
|-------------|-----------------------------|------------------------------------------|
 | `sents`     | `list[list[str, str, ...]]` | A list of lists of sentence-level tokens |
 | `tok_dict`  | `dict{str: int}`            | A dictionary mapping tokens to indexes   |

### <b>Returns:</b>

| Name        | Type                        | Description                                                                 |
|-------------|-----------------------------|-----------------------------------------------------------------------------|
 | `idx_sents` | `list[list[int, int, ...]]` | A list of lists of dictionary-index-mapped representations of each sentence |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `ohe(idx_toks, vocab_length, vtype)`

### <b>Description:</b>

This function forms a matrix of one-hot-encoded
dictionary-index-mapped tokens

### <b>Arguments:</b>

| Name           | Type                        | Description                                             |
|----------------|-----------------------------|---------------------------------------------------------|
 | `idx_toks`     | `list[list[int, int, ...]]` | A dictionary-index-mapped representation of tokens      |
 | `vocab_length` | `int`                       | The number of unique tokens in the token dictionary     |
 | `vtype`        | `str`                       | The type of variable (either "predictor" or "response") |

### <b>Returns:</b>

| Name | Type            | Description                                                                                                           |
|------|-----------------|-----------------------------------------------------------------------------------------------------------------------|
 | `X`  | `numpy.ndarray` | A matrix where every row represents a particular token in a sentence and every column represents a vocabulary feature |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
<br>
<br>

### `preprocess_text.py`

This file makes calls to the functions in `util.py` in order
to load data, preprocess data, write data, and complete other
related tasks. It is the file responsible for transforming the
raw text data into the W2V and OHE vector formats that are
input into the neural network models. As a result,
`preprocess_text.py` is also responsible for instantiating and
training the W2V model used for creating W2V embedding vectors
from the raw text data. Below, documentation of the `main`
function in the file is given.

### `main(ptb_train_file, ptb_dev_file, ptb_test_file)`

### <b>Arguments:</b>

| Name             | Type  | Description                                        |
|------------------|-------|----------------------------------------------------|
 | `ptb_train_file` | `str` | The path to the text file of training sentences    |
 | `ptb_dev_file`   | `str` | The path to the text file of development sentences |
 | `ptb_test_file`  | `str` | The path to the text file of testing sentences     |

### <b>Returns:</b>

This function does not have any returns.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
<br>
<br>

### `train_lstm_model1.py`

This file trains a bidirectional LSTM model using previous
words as predictor variables and the next word as the response
variable. Input words are represented as 100-dimensional W2V
vectors, and each output word is represented as a
10,000-dimensional OHE vector. After training, this file saves
plots of the loss on the training and validation sets, and it
also saves the trained model to an output file. Lastly,
`train_lstm_model1.py` uses the trained model to predict
target words of the testing set vectors by outputting vectors
of probability scores across the entire target vocabulary set.
It saves these vectors to an output file as well. Below,
documentation of the `main` function in the file is given.

### `main(vector_directory, ohe_directory, epochs, units, activation, recurrent_activation, dense_activation, input_shape, model_file)`

### <b>Arguments:</b>

| Name                   | Type    | Description                                                                                                     |
|------------------------|---------|-----------------------------------------------------------------------------------------------------------------|
 | `vector_directory`     | `str`   | The system file path to the word2vec vectors, these are the predictor variables                                 |
 | `ohe_directory`        | `str`   | The system file path to the target variables, represented as OHE words, with dimensionality equal to vocab size |
 | `epochs`               | `int`   | The number of training epochs                                                                                   |
 | `units`                | `int`   | The number of units in first LSTM layer and half the amount of units in the second LSTM layer                   |
 | `activation`           | `str`   | The LSTM layers' activation function                                                                            |
 | `recurrent_activation` | `str`   | The LSTM layers' recurrent activation function                                                                  |
 | `dense_activation`     | `str`   | The activation function of the first dense layer, last layer must be softmax                                    |
 | `input_shape`          | `tuple` | The input shape of the predictor variables (time-steps, features)                                               |
 | `model_file`           | `str`   | The system file path to save the trained model                                                                  |


### <b>Returns:</b>

This function does not have any returns.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
<br>
<br>

### `train_lstm_model2.py`

This file reads preprocessed sentences and converts them to
10,000-dimensional OHE vectors. The file then trains a
bidirectional LSTM model using the OHE representation of the
previous words as predictor variables and the OHE
representation of the next word as the response variable.
However, the model includes an embedding layer to reduce the
input OHE vectors to dense, 100-dimensional vectors prior to
input into the first bidirectional layer. Similar to the first
model, each output word is represented as a 10,000-dimensional
OHE vector. After training, this file saves plots of the loss
on the training and validation sets, and it also saves the
trained model to an output file. Lastly,
`train_lstm_model2.py` uses the trained model to predict
target words of the testing set vectors by outputting vectors
of probability scores across the entire target vocabulary set.
It saves these vectors to an output file as well. Below,
documentation of each function in the file is given.

## Functions:

### `read_data(data_dic)`

### <b>Description:</b>

This function reads the pre-processed sequence data.

### <b>Arguments:</b>

| Name       | Type   | Description                                                                            |
|------------|--------|----------------------------------------------------------------------------------------|
 | `data_dic` | `dict` | Dictionary with keys representing system file paths to the pre-processed sequence data | 

### <b>Returns:</b>

| Name   | Type   | Description                                                                                                                                                                               |
|--------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | `data` | `dict` | The dictionary containing the sequence data where the key is the dataset and the value is the pre-processed numpy array sequence data. Each row is a sentence and each column is a token. |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `word_to_sent(sent)`

### <b>Description:</b>

This function converts a list of tokens to a single string.

### <b>Arguments:</b>

| Name          | Type                  | Description                                                   |
|---------------|-----------------------|---------------------------------------------------------------|
 | `sent`        | `list[str, str, ...]` | The sentence (sequence of tokens) to be converted to a string |

### <b>Returns:</b>

| Name | Type  | Description                                        |
|------|-------|----------------------------------------------------|
 | `s`  | `str` | The single string representation of the sentence   |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `tokens_to_seq(data)`

### <b>Description:</b>

This function uses the keras Tokenizer to convert sequences
of tokens into sequential data ready to be input into model
for training.

### <b>Arguments:</b>

| Name     | Type   | Description                                                                                             |
|----------|--------|---------------------------------------------------------------------------------------------------------|
 | `data`   | `dict` | The data where the keys are the datasets and the values are the pre-processed sentences in token format |

### <b>Returns:</b>

| Name  | Type   | Description                                                                                              |
|-------|--------|----------------------------------------------------------------------------------------------------------|
 | `seq` | `dict` | The data where the keys are the datasets and the values are the sequence format (arrays of word indices) |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `main(ohe_directory, seq_directory, epochs, units, activation, recurrent_activation, dense_activation, model_file)`

### <b>Arguments:</b>

| Name                   | Type    | Description                                                                                                     |
|------------------------|---------|-----------------------------------------------------------------------------------------------------------------|
 | `ohe_directory`        | `str`   | The system file path to the target variables, represented as OHE words, with dimensionality equal to vocab size |
 | `seq_directory`        | `str`   | The system file path to the predictor variables, represented as pre-processed list of tokens                    |
 | `epochs`               | `int`   | The number of training epochs                                                                                   |
 | `units`                | `int`   | The number of units in first LSTM layer and half the amount of units in the second LSTM layer                   |
 | `activation`           | `str`   | The LSTM layers' activation function                                                                            |
 | `recurrent_activation` | `str`   | The LSTM layers' recurrent activation function                                                                  |
 | `dense_activation`     | `str`   | The activation function of the first dense layer, last layer must be softmax                                    |
 | `model_file`           | `str`   | The system file path to save the trained model                                                                  |

### <b>Returns:</b>

This function does not have any returns.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
<br>
<br>

### `train_lstm_model3.py`

This file trains a bidirectional LSTM model using previous
words as predictor variables and the next word as the response
variable. Input and output words are represented as
10,000-dimensional OHE vectors. After training, this file
saves plots of the loss on the training and validation sets,
and it also saves the trained model to an output file. Lastly,
`train_lstm_model3.py` uses the trained model to predict
target words of the testing set vectors by outputting vectors
of probability scores across the entire target vocabulary set.
It saves these vectors to an output file as well. Below,
documentation of the `main` function in the file is given.

### `main(vector_directory, epochs, units, activation, recurrent_activation, dense_activation, input_shape, model_file)`

### <b>Arguments:</b>

| Name                   | Type    | Description                                                                                   |
|------------------------|---------|-----------------------------------------------------------------------------------------------|
 | `vector_directory`     | `str`   | The system file path to the OHE vectors                                                       |
 | `epochs`               | `int`   | The number of training epochs                                                                 |
 | `units`                | `int`   | The number of units in first LSTM layer and half the amount of units in the second LSTM layer |
 | `activation`           | `str`   | The LSTM layers' activation function                                                          |
 | `recurrent_activation` | `str`   | The LSTM layers' recurrent activation function                                                |
 | `dense_activation`     | `str`   | The activation function of the first dense layer, last layer must be softmax                  |
 | `input_shape`          | `tuple` | The input shape of the predictor variables (time-steps, features)                             |
 | `model_file`           | `str`   | The system file path to save the trained model                                                |

### <b>Returns:</b>

This function does not have any returns.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
<br>
<br>

### `evaluate.py`

This file serves to evaluate each of the neural network models
developed in the project. It initiates this process by loading
the W2V model file, the OHE vectors, the predicted probability
vectors from the testing sets, and an array consisting of the
tokens that formed the vocabulary of the Penn Treebank corpus.
The file then assesses the accuracy of each model by checking
the percentage of times the next word predicted was the actual
word. It also assesses accuracy in terms of the number of
times the predicted word was in the top 10 most similar words 
of the actual word (including the actual word). After
assessing these figures, the file generates bar charts of the
results and saves these to the "plots" directory. In addition,
`evaluate.py` has the capability to test out either the W2V
model or the OHE model by predicting what the next word of an
input sentence will be. Below, documentation of each function
in the file is given.

## Functions:

### `make_prediction(sentence, model, word2vec, token_array, input_type)`

### <b>Description:</b>

This function makes a next word prediction using the W2V model
or OHE model of any sentence, given all vocabulary is within
the Penn Treebank corpus.

### <b>Arguments:</b>

| Name          | Type                                      | Description                                                                                       |
|---------------|-------------------------------------------|---------------------------------------------------------------------------------------------------|
 | `sentence`    | `list[str, str, ...]`                     | The list of tokens in the sentence                                                                | 
 | `model`       | `keras.engine.sequential.Sequential`      | The loaded keras model                                                                            |
 | `word2vec`    | `gensim.models.keyedvectors.KeyedVectors` | The word2vec model which should be used to convert tokens to vectors                              | 
 | `token_array` | `list[str, str, ...]`                     | The list of vocabulary tokens, must be in same order as was used to pre-process the original data |
 | `input_type`  | `str`                                     | Which model input to use (either "word2vec" or "ohe")                                             |

### <b>Returns:</b>

| Name    | Type  | Description                     |
|---------|-------|---------------------------------|
 | `token` | `str` | The predicted next word token   |

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>
### `main(w2v_model_file, ohe_vectors, predicted_vectors, token_arr)`

### <b>Description:</b>

This function evaluates the performance of the models.

### <b>Arguments:</b>

| Name                | Type  | Description                                                                                                        |
|---------------------|-------|--------------------------------------------------------------------------------------------------------------------|
 | `w2v_model_file`    | `str` | System file path to the word2vec model                                                                             |
 | `ohe_vectors`       | `str` | System file path to the OHE vector directory                                                                       |
 | `predicted_vectors` | `str` | System file path to the models' prediction directory                                                               |
 | `token_arr`         | `str` | The system file path to the token array, contains all tokens in same order as was used to preprocess original data |

### <b>Returns:</b>

This function does not return anything.

### <b>Raises:</b>

This function does not have any raises.
<br>
<br>

## Other Files

The following is a description of other miscellaneous files
included to help support the project.
<br><br>

### `requirements.txt`

This file documents all the third-party libraries that need to
be installed in order to reproduce the environment that the
project was run in.
<br><br>


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