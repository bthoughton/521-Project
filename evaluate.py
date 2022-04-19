# imports
import argparse
import numpy as np
from gensim.models.word2vec import Word2Vec


def main(w2v_model_file, yp_vectors, y_vectors):

    # load the Word2Vec model
    w2v_mod = Word2Vec.load(w2v_model_file)

    # forms the KeyedVectors token --> vector lookup dictionary
    w2v_vecs = w2v_mod.wv

    # load the predicted and true vectors
    testYp = np.load(yp_vectors)
    testY = np.load(y_vectors)
    trainYp = np.load('data/ohe_vectors/trainYp.npy')
    trainY = np.load('data/ohe_vectors/trainY.npy')

    # Get most common words for predicted and actual words
    # print(w2v_vecs.similar_by_vector(testYp[50]))
    # print(w2v_vecs.similar_by_vector(testY[50]))
    # print(w2v_vecs.similar_by_vector(trainYp[100]))
    # print(w2v_vecs.similar_by_vector(trainY[100]))

    testA = []
    testP = []
    testC = 0
    trainA = []
    trainP = []
    trainC = 0

    for vp, va in zip(testYp, testY):

        testa = w2v_vecs.similar_by_vector(va)[0][0]
        testA.append(testa)

        testp = [word[0] for word in w2v_vecs.similar_by_vector(vp)]
        testP.append(testp)

        if testa in testp:
            testC += 1

    print('The correct word was in testing predictions {} out of {} records'.format(testC, len(testA)))

    for vp, va in zip(trainYp, trainY):

        traina = w2v_vecs.similar_by_vector(va)[0][0]
        trainA.append(traina)

        trainp = [word[0] for word in w2v_vecs.similar_by_vector(vp)]
        trainP.append(trainp)

        if traina in trainp:
            trainC += 1

    print('The correct word was in training predictions {} out of {} records'.format(trainC, len(trainA)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_model_file", type=str,
                        default="models/word2vec.model",
                        help="The trained word2vec model file path")
    parser.add_argument("--yp_vectors", type=str,
                        default="data/ohe_vectors/testYp.npy",
                        help="The predicted word vectors")
    parser.add_argument("--y_vectors", type=str,
                        default="data/ohe_vectors/testY.npy",
                        help="The true word vectors")

    args = parser.parse_args()

    main(args.w2v_model_file, args.yp_vectors, args.y_vectors)

