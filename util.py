# imports
import numpy as np
import pandas as pd
from nltk import word_tokenize


def load_sents(raw_data_path):
    """Converts a newline-separated txt file of sentences to a nested list

    Args:
        raw_data_path (str): The path to the text file of sentences

    Returns:
        sents (list[list[str, str, ...]]): A list of lists of sentence-level
            tokens

    Raises:
        None
    """

    # initialize a list for storing the sentences
    sents = []

    # open the text file for reading
    with open(raw_data_path, 'r', encoding='utf-8') as f:

        # for each sentence in the file
        for sent in f:

            # # if a sentence is present
            # if sent.strip():

            # tokenize the sentence and append it to the main list
            sents.append(word_tokenize(sent))

    # returns the list of sentences
    return sents


def preprocess_sents(sents, fixed_length=False, length=2):
    """Preprocesses a list of sentences. Includes the option of forcing all
       sentences to contain the same number of tokens

    Args:
        sents (list[list[str]]): A list of lists of sentence-level tokens
        fixed_length (bool): A flag denoting whether to fix sentence length
        length (int): The number of tokens that each sentence should include

    Returns:
        list[list[str]]: A list of lists of preprocessed sentence-level tokens

    Raises:
        None
    """

    # initialize a list for storing the lists of preprocessed sentence tokens
    sents_preproc = []

    # initialize a list of tokens to be removed from each sentence
    discards = ['<', 'unk', '>', 'N', '.']

    unk = 0

    # if fixing the length (in tokens) of each sentence is desired
    if fixed_length:

        # for each list of tokens
        for sent in sents:

            # only extract the tokens that are not to be removed
            sent_preproc = [tok for tok in sent if tok not in discards]

            # if the number of tokens matches the required length
            if len(sent_preproc) == length:

                # append the preprocessed sentence list to the main sentence list
                sents_preproc.append(sent_preproc)

            # else, if the number of tokens surpasses the required length
            elif len(sent_preproc) > length:

                # delete all the tokens that cause the length to be surpassed
                del sent_preproc[length:]

                # append the shortened sentence list to the main list
                sents_preproc.append(sent_preproc)

    # if requiring a fixed length for the sentences is not desired
    else:

        # for each list of tokens
        for sent in sents:

            # track the number of sentences with unknown word
            if 'unk' in sent:
                unk += 1

            # only extract the tokens that are not to be removed
            sent_preproc = [tok for tok in sent if tok not in discards]

            # if at least two tokens were returned
            if len(sent_preproc) > 1:

                # append the preprocessed sentence list to the main sentence list
                sents_preproc.append(sent_preproc)

        # print('Sentences with unknown word:', unk)

    # return the preprocessed sentences
    return sents_preproc


def split_sentences(sentences, length):
    """
    Splits sentences into predictor and response variables depending on desired
    length of sentences. The response variable is always the last token to be
    kept, indicated by passed sentence length argument.

    Args:
        sentences (array): A 2-dimensional iterable in which the first dimension
            is each sentence and the 2nd dimension is a list of tokens in the
            sentence, represented as strings.
        length (int): The total number of tokens to keep in the sentence/phrase.
            This value should be the total of the predictor tokens and the
            response token.

    Returns:
        X,Y (tuple): The predictor and response tokens represented as
            2-dimensional iterables in which the first dimension is each
            sentence and the 2nd dimension is a list of tokens in the sentence,
            represented as strings.
    """

    # Instantiate lists to store the variables
    X = []
    Y = []

    # Iterate over all sentences passed to function
    for s in sentences:

        # Check if the length of tokens satisfies the required length argument
        if len(s) >= length:
            # Split the tokens into predictor and response variables
            x = s[:length - 1]
            y = s[length - 1]

            # Add the sentence's tokens to the lists
            X.append(x)
            Y.append(y)

    return X, Y


def write_processed(sentences, fname):
    """
    Saves the preprocessed sentences in token format to csv file after
    converting them to a data frame

    Args:
        sentences (list): A list of lists where the first dimension is a list of
            sentences and the second is a list of tokens in string format.
        fname (str): The system file path to save the processed tokens

    Returns:
        None

    Raises:
        None

    """
    # Convert the list of sentences to a data frame
    df = pd.DataFrame(sentences)

    # Write the data frame to file
    df.to_csv(fname, header=False, index=False, encoding='utf-8')


def word_2_vector(sentences, keyed_vectors, file, v_type):
    """
    Converts list of sentences from word tokens to word vectors. Or converts
    response words to a vector.

    Args:
        sentences (list): A 2-dimensional array where the first dimension is
            each sentence and the 2nd dimension is each token in the sentence.
        keyed_vectors (gensim.models.keyedvectors.KeyedVectors): The word2vec
            model which should be used to convert tokens to vectors.
        file (str): The relative system file path of where to save the sentence
            vectors.
        v_type (str): The type of variable either "predictor" or "response"

    Returns:
        None

    Raises:
        None

    """
    if v_type == 'predictor':
        arr = np.array([[keyed_vectors[tok] for tok in sent] for sent in sentences])

    if v_type == 'response':
        arr = np.array([keyed_vectors[tok] for tok in sentences])

    np.save(file, arr)


def form_dictionary(sents):
    """Uses sentences to form a dictionary mapping sentence-level tokens to
       unique indexes

    Args:
        sents (list[list[str, str, ...]]): A list of lists of sentence-level
            tokens

    Returns:
        tok_dict (dict{str: int}): A dictionary mapping tokens to indexes

    Raises:
        None
    """

    # initialize a dictionary for storing tokens mapped to indexes
    tok_dict = {}

    # for each sentence
    for sent in sents:

        # for each token in the sentence
        for tok in sent:

            # if the token has not already been assigned as a key
            if not (tok in tok_dict):

                # assign the token as a key and give it an index
                tok_dict[tok] = len(tok_dict)

    # return the dictionary mapping tokens to indexes
    return tok_dict


def transform_sents(sents, tok_dict):
    """Uses a token dictionary to form a dictionary-index-mapped representation
       of the tokens in each sentence

    Args:
        sents (list[list[str, str, ...]]): A list of lists of sentence-level
            tokens
        tok_dict (dict{str: int}): A dictionary mapping tokens to indexes

    Returns:
        idx_sents (list[list[int, int, ...]]): A list of lists of
            dictionary-index-mapped representations of each sentence

    Raises:
        None
    """

    # form a list for storing the dict index representation of each sentence
    idx_sents = []

    # for each sentence
    for sent in sents:

        # form a list to store the dict index version of the current sentence
        idx_sent = []

        # Check if the sent is a token (response variables)
        if isinstance(sent, str):

            # If the sentence is
            tok = sent

            # append the token's index to the list
            idx_sent.append(tok_dict[tok])

        else:

            # for each token in the sentence
            for tok in sent:

                # append the token's index to the list
                idx_sent.append(tok_dict[tok])

        # append the dict index version of the sentence to the main list
        idx_sents.append(idx_sent)

    # return the index representation of each sentence
    return idx_sents


def ohe(idx_toks, vocab_length, vtype):
    """Forms a matrix of one-hot-encoded dictionary-index-mapped tokens

    Args:
        idx_toks (list[int, int, ...]): A dictionary-index-mapped
            representation of tokens
        vocab_length (int): The number of unique tokens in the token dictionary
        vtype (str): The type of variable, "X" (predictor) or "Y" (response)

    Returns:
        X (numpy.ndarray): A matrix where every row represents a particular
         token in a sentence and every column represents a vocabulary feature

    Raises:
        None
    """

    if vtype == 'X':
        # initialize a NumPy array to store the one-hot-encoded token(s)
        X = np.zeros((len(idx_toks), vocab_length))

    elif vtype == 'Y':
        # initialize a NumPy array to store the one-hot-encoded token(s)
        Y = np.zeros(vocab_length)

    # for each dictionary-index-mapped token, and its index
    for i, idx_tok in enumerate(idx_toks):

        if vtype == 'X':
            # based on the index, flag the associated vocab feature
            X[i, idx_tok] = 1.

            # return the one-hot-encoded dictionary-index-mapped token matrix
            return X

        elif vtype == 'Y':
            # based on the index, flag the associated vocab feature
            Y[idx_tok] = 1.

            # return the one-hot-encoded dictionary-index-mapped token matrix
            return Y


# test the util functions
if __name__ == '__main__':

    raw_sent = load_sents('data/raw/ptbdataset/ptb.train.txt')
    print(raw_sent[-1])

    processed_sent = preprocess_sents(raw_sent)
    print(processed_sent[-1])

    split_x, split_y = split_sentences(processed_sent, 5)
    print(split_x[-1])
    print(split_y[-1])
    print(len(split_y))

    for s, t in zip(split_x, split_y):
        if 'quack' in t:
            print(s)
            print(t)

