# imports
import argparse
import numpy as np
from gensim.models.word2vec import Word2Vec
from util import (
    form_dictionary,
    load_sents,
    ohe,
    preprocess_sents,
    split_sentences,
    transform_sents,
    word_2_vector,
    write_processed
)


def main(ptb_train_file, ptb_dev_file, ptb_test_file):
    """Loads and preprocesses sentences from the Penn Treebank data set. Uses
       the processed sentences to make a Word2Vec model. Writes the model to an
       output file

    Args:
        ptb_train_file (str): The path to the text file of training sentences
        ptb_dev_file (str): The path to the text file of validation sentences
        ptb_test_file (str): The path to the text file of testing sentences

    Returns:
        None

    Raises:
        None
    """

    print('Loading dataset...')

    # load and tokenize the training sentences
    train_sents = load_sents(ptb_train_file)
    # load and tokenize the validation sentences
    dev_sents = load_sents(ptb_dev_file)
    # load and tokenize the testing sentences
    test_sents = load_sents(ptb_test_file)

    # preprocess the training sentences
    train_processed = preprocess_sents(train_sents)
    # preprocess the validation sentences
    dev_processed = preprocess_sents(dev_sents)
    # preprocess the testing sentences
    test_processed = preprocess_sents(test_sents)

    # combine all the preporcessed sentences
    all_processed_sents = train_processed + dev_processed + test_processed

    # create a Skip-Gram Word2Vec model from the preprocessed sentences
    w2v_mod = Word2Vec(
        sentences=all_processed_sents,  # use all preprocessed sentences
        vector_size=100,                # uses 100-dimensional vectors
        window=5,                       # context window of length 5
        min_count=5,                    # ignores tokens below 5 counts
        sg=1                            # uses skip-gram architecture
    )

    print('Training the word2vec model on dataset...')

    # train the word2vec model
    w2v_mod.train(
        corpus_iterable=all_processed_sents,
        total_examples=w2v_mod.corpus_count,
        epochs=20
    )

    # save the model to an output file
    w2v_mod.save('models/word2vec.model')

    # split the sentences into predictor and response variables
    train_sent_X, train_sent_Y = split_sentences(train_processed, 6)
    dev_sent_X, dev_sent_Y = split_sentences(dev_processed, 6)
    test_sent_X, test_sent_Y = split_sentences(test_processed, 6)

    # form a dictionary of filenames and the sentences they correspond to
    sentences = {
        'trainX': train_sent_X,
        'trainY': train_sent_Y,
        'devX': dev_sent_X,
        'devY': dev_sent_Y,
        'testX': test_sent_X,
        'testY': test_sent_Y,
    }

    print('Splitting data into predictors and response variables...')

    # Write the processed and split sentences to file
    for name, sent in sentences.items():

        # Define the file name
        fname = 'data/processed/' + name + '.csv'

        # Convert to pandas data frame and write to file
        write_processed(sent, fname)

    # Switch to the keyedVector instance
    word_vectors = w2v_mod.wv

    print('Creating word2vec vectors for entire data set...')

    for name, sent in sentences.items():

        # Define the file name
        fname = 'data/vectors/'+name+'.npy'

        # Define the variable type (predictor vs response)
        if 'X' in name:
            v_type = 'predictor'
        else:
            v_type = 'response'

        # Convert tokens to vectors and save to disk
        word_2_vector(sent, word_vectors, fname, v_type)

    # form the one-hot-encoded baseline vectors

    print('Creating the token to word index dictionary...')

    # create a dictionary mapping each processed token to a unique index
    tok_dict = form_dictionary(all_processed_sents)

    # Save the words of the token dictionary as an array
    tok_arr = np.array([word for word in tok_dict.keys()])

    # Save the array
    np.save('data/tok_arr.npy', tok_arr)

    print('Creating the OHE sentence vectors...')

    # for each filename-sentences key-value pair in the dictionary
    for name, sent in sentences.items():

        # Define the file name
        fname = 'data/ohe_vectors/' + name + '.npy'

        # if the variable type is predictor
        if 'X' in name:

            # form a dictionary-index-mapped representation of the tokens
            X_2vec = transform_sents(sent, tok_dict)

            # form a matrix of one-hot-encoded dictionary-index-mapped tokens
            X_ohe = np.array([ohe(sent, len(tok_dict), vtype='X') for sent in X_2vec])

            # save the one-hot-encoded vectors to an output file
            np.save(fname, X_ohe)

        # else, if the variable type is "response"
        else:

            # form a dictionary-index-mapped representation of the tokens
            Y_2vec = transform_sents(sent, tok_dict)

            # form a matrix of one-hot-encoded dictionary-index-mapped tokens
            Y_ohe = np.array([ohe(sent, len(tok_dict), vtype='Y') for sent in Y_2vec])

            # save the one-hot-encoded vectors to an output file
            np.save(fname, Y_ohe)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ptb_train_file", type=str,
                        default="data/raw/ptbdataset/ptb.train.txt",
                        help="raw train file")
    parser.add_argument("--ptb_dev_file", type=str,
                        default="data/raw/ptbdataset/ptb.valid.txt",
                        help="raw validation file")
    parser.add_argument("--ptb_test_file", type=str,
                        default="data/raw/ptbdataset/ptb.test.txt",
                        help="raw test file")
    args = parser.parse_args()

    main(args.ptb_train_file, args.ptb_dev_file, args.ptb_test_file)

