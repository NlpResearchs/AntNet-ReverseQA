import json
import os
from model_utils import train
import tensorflow as tf
import sys
from test import test
import random
import pickle

from data_stream import DataStream
from model import AnswerUnderstander
from utils import read_file, make_instances, \
    make_corpus, generate_vocabulary, \
    split_train_dev, drop_instances_to_excel, \
    gather_question_dict, split_template_choose, gather_samples, \
    gather_targets_for_samples
from bert_encoder import BertEncoder
from w2v_encoder import W2VEncoder

# def train_model():
if __name__ == "__main__":
    with open('config_choose.json', encoding='utf-8') as infile:
        config = json.load(infile)

    int2bool = {1: True, 0: False}

    data_path = config["train_config"]["DATA_PATH"]
    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    max_sequence_len = config["train_config"]["MAX_SEQUENCE_LEN"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_shuffle = int2bool[config["train_config"]["IS_SHUFFLE"]]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    nb_epoch = config["train_config"]["NB_EPOCH"]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    use_bert = int2bool[config["train_config"]["USE_BERT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_train_path = config["train_config"]["DROP_TRAIN_PATH"]
    drop_path = config["train_config"]["DROP_CHOOSE_DEV_PATH"]
    drop_template_path = config["train_config"]["DROP_CHOOSE_TEMPLATE_PATH"]
    best_path = config["train_config"]["BEST_PATH"]
    question2targets_path = config["train_config"]["QUESTION2TARGETS_PATH"]
    use_extra_feature = config["train_config"]["USE_EXTRA_FEATURE"]
    ner_dict_size = config["train_config"]["NER_DICT_SIZE"]
    pos_dict_size = config["train_config"]["POS_DICT_SIZE"]
    extra_feature_dim = config["train_config"]["EXTRA_FEATURE_DIM"]
    ner_dict_path = config["train_config"]["NER_DICT_PATH"]
    pos_dict_path = config["train_config"]["POS_DICT_PATH"]
    rnn_dim = config["train_config"]["RNN_DIM"]
    lambda_l2 = config["train_config"]["LAMBDA_L2"]
    sentiment_polarity_multiple = config["train_config"]["POLARITY_MULTIPLE"]
    pretrained_model = config["train_config"]["PRETRAINED_MODEL"]

    use_w2v = True
    if use_bert:
        use_w2v = False

    char_w2v_path = config["w2v_config"]["CHAR_W2V_PATH"]
    char_voc_path = config["w2v_config"]["CHAR_VOC_PATH"]
    char_embedding_matrix_path = config["w2v_config"][
        "CHAR_EMBEDDING_MATRIX_PATH"]
    word_w2v_path = config["w2v_config"]["WORD_W2V_PATH"]
    word_voc_path = config["w2v_config"]["WORD_VOC_PATH"]
    word_embedding_matrix_path = config["w2v_config"][
        "WORD_EMBEDDING_MATRIX_PATH"]

    bert_model_path = config["bert_config"]["BERT_MODEL_PATH"]
    bert_config_file = config["bert_config"]["CONFIG_FILE"]
    bert_checkpoint_path = config["bert_config"]["INIT_CHECKPOINT"]
    bert_voc_path = config["bert_config"]["VOC_FILE"]
    sen2id_path = config["bert_config"]["SEN2ID_PATH"]

    samples, _, _ = read_file(data_path)
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]
    char_corpus, word_corpus = make_corpus(samples)
    # default sen2id from char_corpus
    sen2id = generate_vocabulary(sen2id_path, char_corpus)
    question2targets = gather_targets_for_samples(samples, True,
                                                  question2targets_path)

    bert_max_seq_len = max([len(sen) for sen in sen2id.keys()]) + 2
    max_sequence_len = max(bert_max_seq_len, max_sequence_len)
    bert_max_seq_len = max_sequence_len

    # question2index = gather_question_dict(samples)
    # question2index, question2template = split_template_choose(samples, question2index)
    # samples_template = gather_samples(samples, question2template)
    # samples = gather_samples(samples, question2index)

    w2v_encoder = W2VEncoder(True, char_corpus, word_corpus,
                             char_w2v_path, char_voc_path,
                             char_embedding_matrix_path, word_w2v_path,
                             word_voc_path, word_embedding_matrix_path)

    bert_encoder = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=bert_max_seq_len,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)

    # ins_template = make_instances(samples_template,
    #                               w2v_encoder.char_voc,
    #                               w2v_encoder.word_voc,
    #                               sentiment_words_path,
    #                               ner_dict_path=ner_dict_path,
    #                               pos_dict_path=pos_dict_path,
    #                               use_extra_feature=use_extra_feature,
    #                               question2targets=question2targets,
    #                               is_training=True,
    #                               need_augment=True)
    instances = make_instances(samples,
                               w2v_encoder.char_voc,
                               w2v_encoder.word_voc,
                               sentiment_words_path,
                               ner_dict_path=ner_dict_path,
                               pos_dict_path=pos_dict_path,
                               use_extra_feature=use_extra_feature,
                               question2targets=question2targets,
                               is_training=True,
                               need_augment=True)
    instances_train, instances_dev = split_train_dev(instances, dev_size=0.2)
    drop_instances_to_excel(instances_train, drop_train_path)
    drop_instances_to_excel(instances_dev, drop_path)
    drop_instances_to_excel(instances, 'temp.xlsx')
    # drop_instances_to_excel(ins_template, drop_template_path)
    instances_train, instances_valid = split_train_dev(instances_train,
                                                       dev_size=0.1)

    # data_stream_train = DataStream(instances=instances_train,
    #                                is_shuffle=is_shuffle,
    #                                is_loop=is_loop,
    #                                batch_size=batch_size,
    #                                ans_max_len=ans_max_len,
    #                                que_max_len=que_max_len,
    #                                use_bert=use_bert,
    #                                bert_encoder=bert_encoder,
    #                                is_sort=is_sort)
    # data_stream_valid = DataStream(instances=instances_valid,
    #                                is_shuffle=False,
    #                                is_loop=is_loop,
    #                                batch_size=batch_size,
    #                                ans_max_len=ans_max_len,
    #                                que_max_len=que_max_len,
    #                                use_bert=use_bert,
    #                                bert_encoder=bert_encoder,
    #                                is_sort=is_sort)

    # with tf.Graph().as_default():
    #     initializer = tf.glorot_uniform_initializer()
    #     global_step = tf.train.get_or_create_global_step()
    #     with tf.variable_scope("Model", reuse=None, initializer=initializer):
    #         answer_understander_train = AnswerUnderstander(
    #             use_bert=use_bert,
    #             use_w2v=use_w2v,
    #             rnn_unit='lstm',
    #             dropout_rate=dropout_rate,
    #             optimizer=optimizer,
    #             learning_rate=learning_rate,
    #             grad_clipper=grad_clipper,
    #             global_step=global_step,
    #             attention_dim=attention_dim,
    #             nb_hops=nb_hops,
    #             rnn_dim=rnn_dim,
    #             lambda_l2=lambda_l2,
    #             is_training=True,
    #             sentiment_polarity_multiple=sentiment_polarity_multiple,
    #             nb_classes=nb_classes,
    #             use_extra_feature=use_extra_feature,
    #             ner_dict_size=ner_dict_size,
    #             pos_dict_size=pos_dict_size,
    #             ans_max_len=ans_max_len,
    #             que_max_len=que_max_len,
    #             extra_feature_dim=extra_feature_dim,
    #             char_w2v_embedding_matrix_path=char_embedding_matrix_path,
    #             word_w2v_embedding_matrix_path=word_embedding_matrix_path)
    #     with tf.variable_scope("Model", reuse=True, initializer=initializer):
    #         answer_understander_valid = AnswerUnderstander(
    #             use_bert=use_bert,
    #             use_w2v=use_w2v,
    #             rnn_unit='lstm',
    #             dropout_rate=dropout_rate,
    #             optimizer=optimizer,
    #             learning_rate=learning_rate,
    #             grad_clipper=grad_clipper,
    #             global_step=None,
    #             attention_dim=attention_dim,
    #             nb_hops=nb_hops,
    #             rnn_dim=rnn_dim,
    #             lambda_l2=lambda_l2,
    #             is_training=False,
    #             sentiment_polarity_multiple=sentiment_polarity_multiple,
    #             nb_classes=nb_classes,
    #             use_extra_feature=use_extra_feature,
    #             ner_dict_size=ner_dict_size,
    #             pos_dict_size=pos_dict_size,
    #             ans_max_len=ans_max_len,
    #             que_max_len=que_max_len,
    #             extra_feature_dim=extra_feature_dim,
    #             char_w2v_embedding_matrix_path=char_embedding_matrix_path,
    #             word_w2v_embedding_matrix_path=word_embedding_matrix_path)
    #     saver = tf.train.Saver()
    #     sess = tf.Session()
    #     initializer = tf.global_variables_initializer()
    #     sess.run(initializer)
    #     print('begin training...')
    #     train(nb_epoch=nb_epoch,
    #           sess=sess,
    #           saver=saver,
    #           data_stream_train=data_stream_train,
    #           data_stream_valid=data_stream_valid,
    #           answer_understander_train=answer_understander_train,
    #           answer_understander_valid=answer_understander_valid,
    #           best_path=best_path)

# if __name__ == "__main__":
#     train_model()
