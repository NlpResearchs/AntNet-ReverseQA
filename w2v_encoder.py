import logging
import os
import pickle
from collections import Counter
import jieba
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec


def build_w2v_matrix(w2v_dict, save_path, voc_path, is_train):
    if is_train:
        embedding_dim = list(w2v_dict.values())[0].shape[0]
        with open(voc_path, 'rb') as infile:
            voc = pickle.load(infile)

        # +1 for padding
        embedding_matrix = np.zeros((len(voc.keys()) + 1, embedding_dim),
                                    dtype='float32')
        for word in voc.keys():
            if word in w2v_dict.keys():
                embedding_matrix[voc[word], :] = w2v_dict[word]
            # random initialize oov
            else:
                embedding_matrix[voc[word], :] = np.random.uniform(
                    -0.05, 0.05, size=(embedding_dim, ))
        logging.info("save the embedding matrix to file")
        with open(save_path, 'wb') as out_file:
            pickle.dump(embedding_matrix, out_file)
    else:
        logging.info("load the embedding matrix from file")
        with open(save_path, 'rb') as infile:
            embedding_matrix = pickle.load(infile)
            embedding_dim = embedding_matrix.shape[1]
    return embedding_matrix, embedding_dim


def generate_vocabulary(word_bag, start_index, voc_path, is_sort=True):
    """
    :param start_index: the specified start index, int
    :param voc_path: the path to save vocabulary generated, str
    :param is_sort: sort word by frequency, bool
    :return: voc: vocabulary, dict
    """
    # samples = read_file(file_path)
    # word_bag = [word for sample in samples for word in jieba.lcut(sample['question'])]
    # word_bag += [word for sample in samples for word in jieba.lcut(sample['answer'])]
    if is_sort:
        word_bag = Counter(word_bag)
        word_bag = sorted(word_bag)
    voc = {}
    for i, word in enumerate(word_bag):
        voc.update({word: i + start_index})
    with open(voc_path, 'wb') as out_file:
        pickle.dump(voc, out_file)
    return voc


class W2VEncoder(object):
    def __init__(self, is_train, char_corpus, word_corpus,
                 char_w2v_path, char_voc_path, char_embedding_matrix_path,
                 word_w2v_path, word_voc_path, word_embedding_matrix_path):
        self.is_train = is_train
        self.char_corpus = char_corpus
        self.word_corpus = word_corpus
        self.char_w2v_path = char_w2v_path
        self.char_voc_path = char_voc_path
        self.char_embedding_matrix_path = char_embedding_matrix_path
        self.word_w2v_path = word_w2v_path
        self.word_voc_path = word_voc_path
        self.word_embedding_matrix_path = word_embedding_matrix_path

        # train and save 2 kinds w2v model
        if self.is_train:
            logging.info("train and save the word2vector model")
            char_w2v_model = Word2Vec(self.char_corpus,
                                      size=128,
                                      window=5,
                                      min_count=1)
            char_w2v_model.train(self.char_corpus,
                                 total_examples=len(char_corpus),
                                 epochs=50)
            char_w2v_model.save(self.char_w2v_path)
            word_w2v_model = Word2Vec(self.word_corpus,
                                      size=128,
                                      window=5,
                                      min_count=1)
            word_w2v_model.train(self.word_corpus,
                                 total_examples=len(word_corpus),
                                 epochs=50)
            word_w2v_model.save(self.word_w2v_path)
        else:
            logging.info("load the saved word2vector model")
            assert os.path.exists(self.char_w2v_path) and os.path.exists(
                self.word_w2v_path)
            char_w2v_model = Word2Vec.load(self.char_w2v_path)
            word_w2v_model = Word2Vec.load(self.word_w2v_path)
        self.char_w2v_model = char_w2v_model
        self.word_w2v_model = word_w2v_model

        gensim_char_dict = Dictionary()
        gensim_char_dict.doc2bow(char_w2v_model.wv.vocab.keys(),
                                 allow_update=True)
        self.char_w2v_dict = {
            char: char_w2v_model[char]
            for _, char in gensim_char_dict.items()
        }
        gensim_word_dict = Dictionary()
        gensim_word_dict.doc2bow(word_w2v_model.wv.vocab.keys(),
                                 allow_update=True)
        self.word_w2v_dict = {
            word: word_w2v_model[word]
            for _, word in gensim_word_dict.items()
        }

        # generate voc
        char_bag = [char for sentence in char_corpus for char in sentence]
        word_bag = [
            word for sentence in char_corpus for word in jieba.lcut(sentence)
        ]
        if self.is_train:
            logging.info("save vocabulary for word2vec model")
            self.char_voc = generate_vocabulary(char_bag,
                                                start_index=1,
                                                voc_path=self.char_voc_path)
            self.word_voc = generate_vocabulary(word_bag,
                                                start_index=1,
                                                voc_path=self.word_voc_path)
        else:
            with open(char_voc_path, 'rb') as infile:
                self.char_voc = pickle.load(infile)
            with open(word_voc_path, 'rb') as infile:
                self.word_voc = pickle.load(infile)

        # build embedding matrix for two model
        self.char_embedding_matrix, self.char_vec_dim = build_w2v_matrix(
            self.char_w2v_dict, self.char_embedding_matrix_path,
            self.char_voc_path, self.is_train)
        self.word_embedding_matrix, self.word_vec_dim = build_w2v_matrix(
            self.word_w2v_dict, self.word_embedding_matrix_path,
            self.word_voc_path, self.is_train)

    def get_char_vec_dim(self):
        return self.char_vec_dim

    def get_word_vec_dim(self):
        return self.word_vec_dim
