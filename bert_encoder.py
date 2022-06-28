import logging
import pickle

import numpy as np
import tensorflow as tf

from bert import modeling
from bert import tokenization
from utils import reverse_dict


def turn_embedding(corpus_vector, max_sequence_len, vec_dim, sen_len):
    """
     distill vector from model's output
    """
    assert len(corpus_vector.shape) == 2

    # corpus_vector[0, :] = zero_vector
    end_corpus_vector = np.zeros([max_sequence_len, vec_dim])
    corpus_vector = corpus_vector[:sen_len + 2, :]
    corpus_vector = extract_embedding(corpus_vector)
    end_corpus_vector[:corpus_vector.shape[0], :] = corpus_vector
    return end_corpus_vector


def extract_embedding(corpus_vector):
    # wipe off representation for [cls] and [sep]
    assert len(corpus_vector.shape) == 2 or len(corpus_vector.shape) == 3
    if len(corpus_vector.shape) == 3:
        return corpus_vector[:, 1: corpus_vector.shape[1] - 1, :]
    return corpus_vector[1: corpus_vector.shape[0] - 1, :]


def load_vocabulary_from_file(voc_path):
    with open(voc_path, 'rb') as infile:
        voc = pickle.load(infile)
    return voc


def prepare_input(corpus, vocab_file, max_sequence_len):
    """
    :return: id,  mask, type for model input
    """
    token = tokenization.CharTokenizer(vocab_file)
    if isinstance(corpus, list):
        word_ids = []
        word_mask = []
        word_segment_ids = []

        for sentence in corpus:
            sen_token = []
            sen_segment_id = []
            sen_token.append("[CLS]")
            sen_segment_id.append(0)
            sen_split_tokens = token.tokenize(sentence)
            for token_ in sen_split_tokens:
                sen_token.append(token_)
                sen_segment_id.append(0)
            sen_token.append("[SEP]")
            sen_segment_id.append(0)
            sen_word_ids = token.convert_tokens_to_ids(sen_token)
            sen_word_mask = [1] * len(sen_word_ids)

            while len(sen_word_ids) < max_sequence_len:
                sen_word_ids.append(0)
                sen_word_mask.append(0)
                sen_segment_id.append(0)

            assert len(sen_word_ids) == max_sequence_len
            assert len(sen_word_mask) == max_sequence_len
            assert len(sen_segment_id) == max_sequence_len

            word_ids.append(sen_word_ids)
            word_mask.append(sen_word_mask)
            word_segment_ids.append(sen_segment_id)
        return np.array(word_ids), np.array(word_mask), np.array(word_segment_ids)

    else:
        assert isinstance(corpus, str)
        split_tokens = token.tokenize(corpus)
        word_token = []
        word_token.append("[CLS]")
        for token_ in split_tokens:
            word_token.append(token_)
        word_token.append("[SEP]")
        word_ids = token.convert_tokens_to_ids(word_token)
        word_mask = [1] * len(word_ids)
        word_segment_ids = [0] * len(word_ids)
        return np.array([word_ids]), np.array([word_mask]), np.array([word_segment_ids])


def make_batches(nb_instances, batch_size):
    nb_batch = int(np.ceil(nb_instances / float(batch_size)))
    batch_index = [(i * batch_size, min(nb_instances, (i + 1) * batch_size)) for i in range(nb_batch)]
    return nb_batch, batch_index


def get_cur_batch_corpus(cur_batch, id2sen):
    return [id2sen[index] for index in range(cur_batch[0], cur_batch[1])]


class BertEncoder(object):

    def __init__(self, model_root, bert_config_file, init_checkpoint, vocab_file, max_sequence_len, embedding_batch,
                 sen2id_path, embedding_matrix_path, vec_dim=768):
        self.model_root = model_root
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.init_checkpoint = init_checkpoint
        self.vocab_file = vocab_file
        # self.save_path = save_path
        self.max_sequence_len = max_sequence_len
        self.embedding_batch = embedding_batch
        self.vec_dim = vec_dim
        self.embedding_matrix_path = embedding_matrix_path
        self.sen2id = load_vocabulary_from_file(sen2id_path)

        # define placeholder
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')  # [batch_size, sentence_length]
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None],
                                         name='input_masks')  # [batch_size, sentence_length]
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None],
                                          name='segment_ids')  # [batch_size, sentence_length]

        # create model
        self.bert_model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )

        # init params from check_point
        tvars = tf.trainable_variables()
        (self.assignment_map, _) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, self.assignment_map)
        self.encoder_out = self.bert_model.get_sequence_output()

    def encode_by_model(self, corpus, save_path=None):
        """
        :param corpus: data to be encoded
        :param sen2id_path: save the sen2id voc
        :param save_path: the path save the embedding matrix pickle file for training data
        :return:
        """

        with tf.Session() as sess:
            # fd = {self.input_ids: np.array(word_ids), self.input_mask: np.array(word_mask),
            #       self.segment_ids: np.array(word_segment_ids)}
            # fd = {self.input_ids: np.array([[2, 44, 55], [3, 5, 344]]),
            #       self.input_mask: np.array([[1, 1, 1], [1, 1, 1]]), self.segment_ids: np.array([[0,0,0],[0,0,0]])}
            sess.run(tf.global_variables_initializer())

            if isinstance(corpus, str):
                word_ids, word_mask, word_segment_ids = prepare_input(corpus, self.vocab_file, self.max_sequence_len)
                feed_dict = {self.input_ids: np.array(word_ids), self.input_mask: np.array(word_mask),
                             self.segment_ids: np.array(word_segment_ids)}
                corpus_vector = sess.run(self.encoder_out, feed_dict)
                corpus_vector = extract_embedding(corpus_vector)
                print('last shape:{}'.format(corpus_vector.shape))
            else:
                assert isinstance(corpus, list)
                id2sen = reverse_dict(self.sen2id)
                assert len(self.sen2id) == len(id2sen)
                # print(sen2id)
                nb_batch, batch_indexs = make_batches(len(corpus), self.embedding_batch)
                bert_embedding_matrix = np.zeros([len(corpus), self.max_sequence_len, self.vec_dim])
                # print(nb_batch)
                # print(batch_indexs)
                for batch_index in range(nb_batch):
                    if batch_index % 500 == 0:
                        logging.info("'current bert embedding_batch index:{}'.format(batch_index)")
                    cur_batch = batch_indexs[batch_index]
                    cur_corpus = get_cur_batch_corpus(cur_batch, id2sen)
                    cur_word_ids, cur_word_mask, cur_word_segment_ids = prepare_input(cur_corpus, self.vocab_file,
                                                                                      self.max_sequence_len)
                    feed_dict = {self.input_ids: cur_word_ids, self.input_mask: cur_word_mask,
                                 self.segment_ids: cur_word_segment_ids}
                    cur_corpus_vector = sess.run(self.encoder_out, feed_dict)
                    # print(cur_corpus_vector)
                    assert len(cur_corpus_vector.shape) == 3
                    for sen_index in range(cur_corpus_vector.shape[0]):
                        bert_embedding_matrix[self.sen2id[cur_corpus[sen_index]], :, :] = turn_embedding(
                            cur_corpus_vector[sen_index, :, :], self.max_sequence_len, self.vec_dim,
                            len(cur_corpus[sen_index]))
                        # print(turn_embedding(cur_corpus_vector[sen_index, :, :], self.max_sequence_len, self.vec_dim,
                        #       len(cur_corpus[sen_index])))
                if save_path:
                    with open(save_path, 'wb') as outfile:
                        pickle.dump(bert_embedding_matrix, outfile)
                print('last shape:{}'.format(bert_embedding_matrix.shape))
                return bert_embedding_matrix
