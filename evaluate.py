# imports
import argparse
import numpy as np
from pathlib import Path
from gensim.models.word2vec import Word2Vec


def main(w2v_model_file, ohe_vectors, predicted_vectors, token_arr):

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

