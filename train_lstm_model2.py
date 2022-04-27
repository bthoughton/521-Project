import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Embedding
from keras.callbacks import ModelCheckpoint


# Define function to read in data
def read_data(data_dic):
    """
    Reads the pre-processed sequence data
    Args:
        data_dic (dict): Dictionary with keys representing system file paths
            to the pre-processed sequence data.

    Returns:
        data (dict): The dictionary containing the sequence data where the key
            is the dataset and the value is the pre-processed numpy array
            sequence data. Each row is a sentence and each column is a token.

    Raises:
        None

    """

    data = data_dic

    # Iterate over keys in data dict
    for key in data.keys():

        # Define file path
        fname = 'data/processed/' + key + '.csv'

        # Read data into python as pandas dataframe
        data[key] = pd.read_csv(fname, header=None, index_col=False).to_numpy()

        # Get the shape
        shape = data[key].shape

        # Define message for logger
        msg = 'Read {} with shape {}'.format(fname, shape)

        # # Log message
        # logging.info(msg)
        print(msg)

    return data


def word_to_sent(sent):
    """
    Convert list of tokens to a single string.

    Args:
        sent (iterable): The sentence (sequence of tokens) to be converted to a
            string.

    Returns:
        s (str): The single string representation of the sentence.

    Raises:
        None
    """

    s = ''
    for tok in sent:
        s += ' ' + tok

    return s


def tokens_to_seq(data):
    """
    Uses the keras Tokenizer to convert sequences of tokens into sequential
    data ready to be input into model for training.

    Args:
        data (dict): The data where the keys are the datasets and the values are
            the pre-processed sentences in token format.

    Returns:
        seq (dict): The data where the keys are the datasets and the values are
            the sequence format (arrays of word indices).

    Raises:
        None

    """

    # Iterate over all 6 data sets
    for name, arr in data.items():

        # Convert all the token lists to sentences with whitespace seperator
        data[name] = np.apply_along_axis(word_to_sent, 1, arr)

    all_sent = np.concatenate((data['trainX'], data['devX'], data['testX']))
    print(all_sent.shape)

    # Instantiate a word tokenizer
    tokenizer = Tokenizer(num_words=10000)

    # Train the tokenizer
    tokenizer.fit_on_texts(all_sent)

    # Save the word index
    word_index = tokenizer.word_index

    # Create the sequences of tokenized words
    seq = {name: pad_sequences(np.array(tokenizer.texts_to_sequences(arr), dtype=object)) for name, arr in data.items()}

    # Pad all sequences
    seq = {name: pad_sequences(arr, maxlen=10) for name, arr in seq.items()}

    return seq


def main(
        ohe_directory,
        seq_directory,
        epochs,
        units,
        activation,
        recurrent_activation,
        dense_activation,
        model_file
):
    """
    Trains a bidirectional LSTM model using previous words as predictor
    variables and the next word as the response variable. The input is
    represented as sequential word indices and the and the output is the
    probability for each word being the next word in the sentence. This model
    trains its own 100 dimensional embedding space.

    Args:
        ohe_directory (str): The system file path to the target variables,
            represented as OHE words, with dimensionality equal to vocab size.
        seq_directory (str): The system file path to the predictor variables,
            represented as pre-processed list of tokens.
        epochs (int): The number of training epochs.
        units (int): The number of units in first LSTM layer and half the amount
            of units in the second LSTM layer.
        activation (str): The LSTM layers' activation function.
        recurrent_activation (str): The LSTM layers' recurrent activation
            function.
        dense_activation (str): The activation function of the first dense
            layer, last layer must be softmax.
        model_file (str): The system file path to save the trained model.

    Returns:
        None

    Raises:
        None
    """

    # Instantiate empty dic to store sequence data
    seq = {
        'trainX': [],
        'devX': [],
        'testX': []
    }

    # Iterate over the files in the sequence directory (predictor variables)
    for f in Path(seq_directory).iterdir():

        # Check if the file is a predictor variable
        if 'X' in f.name:

            print(f)

            # Define the data's key
            key = f.name.replace('.csv', '')

            # Read in the processed text data
            seq[key] = pd.read_csv(f, header=None, index_col=False, encoding='utf-8').to_numpy()

            # Get the data's shape
            shape = seq[key].shape

            print('Read file {} with shape {}'.format(f, shape))

    # Convert the sentences to sequence of word indices
    seq = tokens_to_seq(seq)

    # Instantiate Path object for word vector directory
    ohe_dir = Path(__file__).parent.joinpath(ohe_directory)

    # Instantiate dictionary to store the target OHE vectors
    ohe = {}

    for f in ohe_dir.iterdir():

        if 'Y' in f.name:
            print('Loading {}'.format(f.name))

            # Define the key for the array
            key = f.name.replace('.npy', '')

            # Load the word vector array and place into dictionary
            ohe[key] = np.load(str(f))

            # Define the data's shape
            shape = ohe[key].shape

            print('Read file {} with shape {}'.format(f, shape))

    # Define the data set vocabulary size
    vocab_size = shape[1]

    print('The vocabulary size is {}'.format(vocab_size))

    # Check if gpu is being used for training
    if tf.config.list_physical_devices('GPU'):
        msg = '#################### Using GPU for training ####################'
        print(msg)

    ###########################################################################
    # BUILD MODEL
    ###########################################################################

    # Specify sequential model stack
    model = Sequential()

    model.add(
        Embedding(
            input_dim=10000,
            output_dim=100)
    )

    # Add LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
                recurrent_regularizer=regularizers.l1_l2(0.00, 0.00),
                bias_regularizer=regularizers.l1_l2(0.00, 0.00),
                dropout=0.25,
                recurrent_dropout=0.25,
                return_sequences=True
            )
        )
    )

    # Add LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                units=units*2,
                activation=activation,
                recurrent_activation=recurrent_activation,
                kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
                recurrent_regularizer=regularizers.l1_l2(0.00, 0.00),
                bias_regularizer=regularizers.l1_l2(0.00, 0.00),
                dropout=0.25,
                recurrent_dropout=0.25,
                return_sequences=False
            )
        )
    )
    # Add dense layer
    model.add(
        Dense(
            units=200,
            activation=dense_activation,
            kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
            bias_regularizer=regularizers.l1_l2(0.00, 0.00)
        )
    )

    # Add dense layer
    model.add(
        Dense(
            units=vocab_size,
            activation='softmax',
            kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
            bias_regularizer=regularizers.l1_l2(0.00, 0.00)
        )
    )

    # Get model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    # Define callback to save best model based upon validation mae
    model_checkpoint = ModelCheckpoint(
        filepath=model_file,
        save_weights_only=False,
        monitor='val_acc',
        mode='auto',
        save_best_only=True
    )

    # Train the model
    history = model.fit(
        x=seq['trainX'],
        y=ohe['trainY'],
        epochs=epochs,
        validation_data=(seq['devX'], ohe['devY']),
        batch_size=128,
        callbacks=[model_checkpoint]
    )

    # Get the mse and acc for training and validation data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Define the epoch range
    epochs = range(1, len(loss) + 1)

    # Define a plot figure with 4 subplots
    fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, dpi=300)

    # Adjust the plot spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Define the mae plot
    ax[0].plot(epochs, loss, 'bo', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epoch Number')
    ax[0].set_ylabel('Loss (Categorical Cross-Entropy)')
    ax[0].legend()

    # Define the loss plot
    ax[1].plot(epochs, acc, 'bo', label='Training ACC')
    ax[1].plot(epochs, val_acc, 'b', label='Validation ACC')
    ax[1].set_title('Training and Validation ACC')
    ax[1].set_xlabel('Epoch Number')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    # Save the plot
    plt.savefig('plots/loss2.png')

    test_yp = model.predict(seq['testX'])
    np.save('data/predicted/model2.npy', test_yp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ohe_directory",
        type=str,
        default="data/ohe_vectors",
        help="OHE word vector directory system file path"
    )
    parser.add_argument(
        "--seq_directory",
        type=str,
        default="data/processed",
        help="Pre-processed sequence directory system file path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="The number of training epochs"
    )
    parser.add_argument(
        "--units",
        type=int,
        default=100,
        help="The number of LSTM units"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default='tanh',
        help="The LSTM activation function"
    )
    parser.add_argument(
        "--recurrent_activation",
        type=str,
        default='sigmoid',
        help="The LSTM recurrent activation function"
    )
    parser.add_argument(
        "--dense_activation",
        type=str,
        default='relu',
        help="The first dense layer activation function"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default='models/LSTM2.h5',
        help="Path to save model"
    )
    args = parser.parse_args()

    main(
        args.ohe_directory,
        args.seq_directory,
        args.epochs,
        args.units,
        args.activation,
        args.recurrent_activation,
        args.dense_activation,
        args.model_file
    )
