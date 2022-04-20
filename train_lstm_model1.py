# imports
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
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
        vectors[key] = np.load(str(file))

        print('The vector shape is {}\n'.format(vectors[key].shape))

    # Check if gpu available
    if tf.config.list_physical_devices('GPU'):
        # Define message for logger
        msg = '##### Using GPU for training #####'
        # Log message
        #logger.info(msg)
        print(msg)

    # Specify sequential model stack
    model = Sequential()

    # Add LSTM layer
    model.add(
        LSTM(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            #kernel_regularizer=regularizers.l1_l2(0.001, 0.001),
            #recurrent_regularizer=regularizers.l1_l2(0.001, 0.001),
            #bias_regularizer=regularizers.l1_l2(0.001, 0.001),
            #dropout=P['DO'],
            #recurrent_dropout=P['RD'],
            return_sequences=True,
            input_shape=input_shape
        )
    )

    # Add LSTM layer
    model.add(
        LSTM(
            units=units*2,
            activation=activation,
            recurrent_activation=recurrent_activation,
            #kernel_regularizer=regularizers.l1_l2(0.001, 0.001),
            #recurrent_regularizer=regularizers.l1_l2(0.001, 0.001),
            #bias_regularizer=regularizers.l1_l2(0.001, 0.001),
            #dropout=P['DO'],
            #recurrent_dropout=P['RD'],
            # return_sequences=True,
            input_shape=input_shape
        )
    )
    # Add dense layer
    model.add(
        Dense(
            units=1000,
            activation=dense_activation,
            #kernel_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
            #bias_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
        )
    )

    # Add dense layer
    model.add(
        Dense(
            units=100,
            activation=dense_activation,
            #kernel_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
            #bias_regularizer=regularizers.l1_l2(P['L1'], P['L2']),
        )
    )

    # Get model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer='rmsprop',
        loss='mse'
    )

    # Define callback to save best model based upon validation mae
    model_checkpoint = ModelCheckpoint(
        filepath=model_file,
        save_weights_only=False,
        monitor='val_loss',
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

    # Define the epoch range
    epochs = range(1, len(loss) + 1)

    # Define a plot figure with 4 subplots
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Define the mae plot
    ax.plot(epochs, loss, 'bo', label='Training Loss')
    ax.plot(epochs, val_loss, 'b', label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss (Mean Squared Error)')
    ax.legend()

    # Save and show the plot
    plt.savefig('plots/loss1.png')
    #plt.show()

    testYp = model.predict(vectors['testX'])
    trainYp = model.predict(vectors['trainX'])
    np.save('data/vectors/testYp.npy', testYp)
    np.save('data/vectors/trainYp.npy', trainYp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vector_directory",
        type=str,
        default="data/vectors",
        help="word vector directory relative file path"
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
        default=32,
        help="The number of LSTM units"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default='relu',
        help="The LSTM activation function"
    )
    parser.add_argument(
        "--recurrent_activation",
        type=str,
        default='tanh',
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
        default=(5, 100),
        help="The input shape, (time-steps, features)"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default='models/LSTM.h5',
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

