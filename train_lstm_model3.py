# imports
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout
from keras.callbacks import ModelCheckpoint


def main(
        vector_directory,
        epochs,
        units,
        activation,
        recurrent_activation,
        dense_activation,
        input_shape,
        model_file
):

    # Instantiate Path object for word vector directory
    v_dir = Path(__file__).parent.joinpath(vector_directory)

    # Instantiate dictionary to store the word vector arrays
    vectors = {}

    for file in v_dir.iterdir():

        print('Loading {}'.format(file.name))

        # Define the key for the array
        key = file.name.replace('.npy', '')

        # Load the word vector array and place into dictionary
        vectors[key] = np.load(str(file), allow_pickle=True)

        print('The vector shape is {}\n'.format(vectors[key].shape))

    vocab_size = vectors[key].shape[-1]
    print(vocab_size)

    # Check if gpu available
    if tf.config.experimental.list_physical_devices('GPU'):
        # Define message for logger
        msg = '##### Using GPU for training #####'
        # Log message
        # logger.info(msg)
        print(msg)

    # Define the input shape
    # input_shape = vectors

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
                recurrent_dropout=0.25
            )
        )
    )
    # Add dense layer
    model.add(
        Dense(
            units=200,
            activation=dense_activation,
            # kernel_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
            # bias_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
        )
    )

    # Add dense layer
    model.add(
        Dense(
            units=vocab_size,
            activation='softmax',
            # kernel_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
            # bias_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
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
        x=vectors['trainX'],
        y=vectors['trainY'],
        epochs=epochs,
        validation_data=(vectors['devX'], vectors['devY']),
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

    # Define the mse plot
    ax[0].plot(epochs, loss, 'bo', label='Training Loss')
    ax[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].set_xlabel('Epoch Number')
    ax[0].set_ylabel('Loss (Categorical Cross-Entropy)')
    ax[0].legend()

    # Define the accuracy plot
    ax[1].plot(epochs, acc, 'bo', label='Training ACC')
    ax[1].plot(epochs, val_acc, 'b', label='Validation ACC')
    ax[1].set_title('Training and Validation ACC')
    ax[1].set_xlabel('Epoch Number')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    # Save and show the plot
    plt.savefig('plots/loss3.png')
    #plt.show()

    # make model predictions and save
    testYp = model.predict(vectors['testX'])
    np.save('data/ohe_vectors/testYp.npy', testYp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vector_directory",
        type=str,
        default="data/ohe_vectors",
        help="word vector directory relative file path"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
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
        help="The LSTM recurrent activation function"
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(5, 9999),
        help="The input shape, (time-steps, features)"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default='models/LSTM3.h5',
        help="Path to save model"
    )
    args = parser.parse_args()

    main(
        args.vector_directory,
        args.epochs,
        args.units,
        args.activation,
        args.recurrent_activation,
        args.dense_activation,
        args.input_shape,
        args.model_file
    )

