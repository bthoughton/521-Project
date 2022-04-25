# imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util import transform_sents, ohe
from pathlib import Path
from gensim.models.word2vec import Word2Vec
from keras.models import load_model


def make_prediction(sentence, model, word2vec, token_array, input_type):
    """
    Makes a next word prediction using the Word2Vec model of OHE model of any
    sentence given all vocabulary is within the Penn Treebank corpus.

    Arguments:
        sentence (list): The list of tokens in the sentence.
        model (keras.engine.sequential.Sequential): The loaded keras model.
        word2vec (gensim.models.keyedvectors.KeyedVectors): The word2vec model
            which should be used to convert tokens to vectors.
        token_array (dict): The list of vocabulary tokens, mapping to their
            word indices.
        input_type (str): Which model input to use, either "word2vec" or "ohe".

    Returns:
        token (str): The predicted next word token

    Raises:
        None
    """

    # Check the model input type
    if input_type == 'word2vec':
        # Convert tokens to word2vecc
        mod_inp = np.array([[word2vec[word] for word in sentence]])

    elif input_type == 'ohe':
        # Define the OHE token array
        token_to_idx = {token: idx for idx, token in enumerate(token_array)}

        # Convert tokens to OHE vector
        mod_inp = np.array(
            [ohe(transform_sents(sentence, token_to_idx), len(token_to_idx), 'X')]
        )

    # Make next word prediction with model
    prediction = model.predict(mod_inp)

    # Get the word index with the highest probability
    prediction_idx = np.argmax(prediction)

    # Convert next word from vector to token
    token = token_array[prediction_idx]
    print(token)

    return token


def main(w2v_model_file, ohe_vectors, predicted_vectors, token_arr):
    """
    Evaluates the performance of the models.

    Args:
        w2v_model_file (str): System file path to the word2vec model.
        ohe_vectors (str): System file path to the OHE vector directory.
        predicted_vectors (str): System file path to the models' prediction
            directory.
        token_arr (str): The system file path to the token array, contains all
            tokens in same order as was used to preprocess original data.

    Returns:
        None

    Raises:
        None
    """

    # load the Word2Vec model
    w2v_mod = Word2Vec.load(w2v_model_file)

    # forms the KeyedVectors token --> vector lookup dictionary
    w2v_vecs = w2v_mod.wv

    # Define path to the ohe and predicted vectors
    ohe_dir = Path(__file__).parent.joinpath(ohe_vectors)
    pre_dir = Path(__file__).parent.joinpath(predicted_vectors)

    # Load the token array
    tok_arr = np.load(token_arr)

    # Define dictionary to store vectors
    vectors = {
        'testY': None,
        'model1': None,
        'model2': None,
        'model3': None
    }

    # Iterate over the files in the ohe directory
    for f in ohe_dir.iterdir():

        # Check if the file name is desired dictionary key list
        if f.name.replace('.npy', '') in vectors.keys():

            # Load the vector
            vectors[f.name.replace('.npy', '')] = np.load(f)

    # Iterate over the files in the predictions directory
    for f in pre_dir.iterdir():

        # Check if the file name is desired dictionary key list
        if f.name.replace('.npy', '') in vectors.keys():

            # Load the vector
            vectors[f.name.replace('.npy', '')] = np.load(f)

    # Define the actual word idx and actual word
    act_idx = np.argmax(vectors['testY'], axis=1)

    # Define the actual word array
    act_word = tok_arr[act_idx]

    # Get the top 9 most similar words for the actual word and add the actual
    # word into 2 dimensional list
    top_ten = [
        [sim_word[0] for sim_word in w2v_vecs.similar_by_key(word, topn=9)] + [word]
        for word in list(act_word)
    ]

    # Instantiate list to store model names, scores and word in top ten scores
    labels = []
    correct_scores = []
    top_ten_scores = []

    # Iterate over the vectors
    for name in vectors.keys():

        # check if the vector is a prediction or actual vector
        if vectors[name] is not None and name != 'testY':

            # Define the predicted word index and actual word
            pre_idx = np.argmax(vectors[name], axis=1)
            pre_word = tok_arr[pre_idx]
            records = act_idx.shape[0]
            correct = np.where(act_idx == pre_idx, 1, 0).sum()
            acc = correct/records
            print('Found a total of {} records'.format(records))
            print('The test accuracy of model {} is {:.3f}%'.format(name[-1], acc*100))

            # Begin a predicted in top ten counter
            success = 0

            # Iterate over every top ten record set and prediction
            for word, sim_words in zip(pre_word, top_ten):

                # Check if the prediction is in the top 10 most similar words
                if word in sim_words:
                    success += 1

            print(
                'The number of times the predicted word was in the top 10 most similar words was {} ({:.3f}%)\n'
                .format(success, 100*success/records)
            )

            labels.append(name)
            correct_scores.append(correct)
            top_ten_scores.append(success)

    # Convert model label names to actual type of model
    model_types = {
        'model1': 'Word2Vec Embeddings',
        'model2': 'Trained Embeddings',
        'model3': 'OHE Only Input'
    }

    labels = [model_types[label] for label in labels]

    # Define the bar chart x-axis
    x_axis = np.arange(len(labels))

    # Define a plot figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Define the bar chart bars
    ax.bar(x_axis - 0.2, correct_scores, width=0.4, label='Correct Word')
    ax.bar(x_axis + 0.2, top_ten_scores, width=0.4, label='Word in Top 10')

    ax.set_title('Accuracy Results')
    ax.set_ylabel('Number of Correct Words')

    # Define the x-axis tick marks as model names
    ax.set_xticks(x_axis, labels)

    # Show the plot legend
    ax.legend()

    # Save plot
    plt.savefig('plots/acc_results.png')

    # Load model 1 & 3
    model1 = load_model('models/LSTM1.h5')
    model3 = load_model('models/LSTM3.h5')

    # Define made up tokens to predict next word
    tokens = ['what', 'time', 'are', 'you', 'going']

    # Get the next word prediction of model 1
    next_word1 = make_prediction(
        tokens,
        model1,
        w2v_vecs,
        tok_arr,
        'word2vec'
    )

    # Get the next word prediction of model 3
    next_word3 = make_prediction(
        tokens,
        model3,
        w2v_vecs,
        tok_arr,
        'ohe'
    )

    print('Model 1 Prediction: {}'.format(next_word1))
    print('Model 3 Prediction: {}'.format(next_word3))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_model_file", type=str,
                        default="models/word2vec.model",
                        help="The trained word2vec model file path")
    parser.add_argument("--ohe_vectors", type=str,
                        default="data/ohe_vectors",
                        help="The ohe word vectors")
    parser.add_argument("--predicted_vectors", type=str,
                        default="data/predicted",
                        help="The predicted ohe vectors")
    parser.add_argument("--token_arr", type=str,
                        default="data/tok_arr.npy",
                        help="The token index array file path")

    args = parser.parse_args()

    main(
        args.w2v_model_file,
        args.ohe_vectors,
        args.predicted_vectors,
        args.token_arr
    )

