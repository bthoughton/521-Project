import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from keras.callbacks import ModelCheckpoint


def main(
        vector_directory,
        ohe_directory,
        epochs,
        units,
        activation,
        recurrent_activation,
        dense_activation,
        input_shape,
        model_file
):
    """
    Trains a bidirectional LSTM model using previous words as predictor
    variables and the next word as the response variable. The input is 100
    dimensional word2vec vectors and the output is the probability for each
    word being the next word in the sentence.

    Args:
        vector_directory (str): The system file path to the word2vec vectors,
            these are the predictor variables.
        ohe_directory (str): The system file path to the target variables,
            represented as OHE words, with dimensionality equal to vocab size.
        epochs (int): The number of training epochs.
        units (int): The number of units in first LSTM layer and half the amount
            of units in the second LSTM layer.
        activation (str): The LSTM layers' activation function.
        recurrent_activation (str): The LSTM layers' recurrent activation
            function.
        dense_activation (str): The activation function of the first dense
            layer, last layer must be softmax.
        input_shape (tuple): The input shape of the predictor variables
            (time-steps, features)
        model_file (str): The system file path to save the trained model.

    Returns:
        None

    Raises:
        None
    """

    # Instantiate Path object for word vector directory
    v_dir = Path(__file__).parent.joinpath(vector_directory)
    ohe_dir = Path(__file__).parent.joinpath(ohe_directory)

    # Instantiate dictionary to store the word vector arrays
    vectors = {}

    # Iterate over the word2vec vector files
    for file in v_dir.iterdir():

        # Check if vector is predictor variable
        if 'X' in file.name:
            print('Loading {}'.format(file.name))

            # Define the key for the array
            key = file.name.replace('.npy', '')

            # Load the word vector array and place into dictionary
            vectors[key] = np.load(str(file))

            print('The vector shape is {}\n'.format(vectors[key].shape))

    # Iterate over the ohe vector files
    for file in ohe_dir.iterdir():

        if 'Y' in file.name:
            print('Loading {}'.format(file.name))

            # Define the key for the array
            key = file.name.replace('.npy', '')

            # Load the word vector array and place into dictionary
            vectors[key] = np.load(str(file))

            print('The vector shape is {}\n'.format(vectors[key].shape))

    # Define the vocab size
    vocab_size = vectors[key].shape[1]

    # Check if gpu available
    if tf.config.list_physical_devices('GPU'):
        # Define message for logger
        msg = '##### Using GPU for training #####'
        print(msg)

    # Specify sequential model stack
    model = Sequential()

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
            ),
            input_shape=input_shape
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
            units=128,
            activation=dense_activation,
            kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
            bias_regularizer=regularizers.l1_l2(0.00, 0.00),
        )
    )

    # Add the final dense layer (outputs probability for each word in vocab)
    model.add(
        Dense(
            units=vocab_size,
            activation='softmax',
            kernel_regularizer=regularizers.l1_l2(0.00, 0.00),
            bias_regularizer=regularizers.l1_l2(0.00, 0.00),
        )
    )

    # Get model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer='rmsprop',
        metrics=['acc'],
        loss='categorical_crossentropy'
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
        x=vectors['trainX'],
        y=vectors['trainY'],
        epochs=epochs,
        validation_data=(vectors['devX'], vectors['devY']),
        batch_size=128,
        callbacks=[model_checkpoint]
    )

    # Get the mse and mae for training data and validaiton data
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

    # Define the loss plot
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

    # Save and show the plot
    plt.savefig('plots/loss1.png')

    # Predict the test data and write to file
    testYp = model.predict(vectors['testX'])
    np.save('data/predicted/model1.npy', testYp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vector_directory",
        type=str,
        default="data/vectors",
        help="word vector directory relative file path"
    )
    parser.add_argument(
        "--ohe_directory",
        type=str,
        default="data/ohe_vectors",
        help="ohe vector directory relative file path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        help="The activation function of the first dense layer"
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(5, 100),
        help="The input shape, (time-steps, features)"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default='models/LSTM1.h5',
        help="Path to save model"
    )
    args = parser.parse_args()

    main(
        args.vector_directory,
        args.ohe_directory,
        args.epochs,
        args.units,
        args.activation,
        args.recurrent_activation,
        args.dense_activation,
        args.input_shape,
        args.model_file
    )

