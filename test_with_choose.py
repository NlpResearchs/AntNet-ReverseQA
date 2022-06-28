import json
import pickle
import tensorflow as tf
import numpy as np
from model_utils import evaluation
from data_stream import DataStream
from model import AnswerUnderstander
from bert_encoder import BertEncoder
import matplotlib.pyplot as plt
# from lstm_encoder import LSTMENCODER
from utils import read_file, make_instances, gather_targets_for_samples, \
    gather_question_dict, gather_question_template


def test():
    with open('config_choose.json', encoding='utf-8') as infile:
        config = json.load(infile)

    int2bool = {1: True, 0: False}

    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    drop_template_path = config["train_config"]["DROP_choose_TEMPLATE_PATH"]
    use_bert = int2bool[config["train_config"]["USE_BERT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_choose_dev_path = config["train_config"]["DROP_choose_DEV_PATH"]
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
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]
    sentiment_polarity_multiple = config["train_config"]["POLARITY_MULTIPLE"]
    use_w2v = True
    if use_bert:
        use_w2v = False

    char_voc_path = config["w2v_config"]["CHAR_VOC_PATH"]
    char_embedding_matrix_path = config["w2v_config"][
        "CHAR_EMBEDDING_MATRIX_PATH"]
    word_voc_path = config["w2v_config"]["WORD_VOC_PATH"]
    word_embedding_matrix_path = config["w2v_config"][
        "WORD_EMBEDDING_MATRIX_PATH"]

    bert_model_path = config["bert_config"]["BERT_MODEL_PFTH"]
    bert_config_file = config["bert_config"]["CONFIG_FILE"]
    bert_checkpoint_path = config["bert_config"]["INIT_CHECKPOINT"]
    bert_voc_path = config["bert_config"]["VOC_FILE"]
    sen2id_path = config["bert_config"]["SEN2ID_PATH"]

    choose_samples, _, _ = read_file(drop_choose_dev_path)
    # choose_template, _, _ = read_file(drop_template_path)
    max_sequence_len = max(
        max([len(sample['question']) for sample in choose_samples]),
        max([len(sample['answer']) for sample in choose_samples]))
    with open(char_voc_path, 'rb') as infile:
        char_voc = pickle.load(infile)
    with open(word_voc_path, 'rb') as infile:
        word_voc = pickle.load(infile)

    bert_encoder = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=max_sequence_len,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)

    instances_choose_dev = make_instances(
        choose_samples,
        char_voc,
        word_voc,
        sentiment_words_path,
        ner_dict_path=ner_dict_path,
        pos_dict_path=pos_dict_path,
        use_extra_feature=use_extra_feature,
        question2targets=question2targets,
        is_training=False,
        need_augment=False)

    # instances_choose_dev_with_match_result(instances_choose_dev)

    data_stream_choose_dev = DataStream(instances=instances_choose_dev,
                                        is_shuffle=False,
                                        is_loop=is_loop,
                                        batch_size=batch_size,
                                        ans_max_len=ans_max_len,
                                        que_max_len=que_max_len,
                                        use_bert=use_bert,
                                        bert_encoder=bert_encoder,
                                        is_sort=is_sort)
    with tf.Graph().as_default():
        with tf.variable_scope("Model", reuse=False,
                               initializer=tf.glorot_uniform_initializer()):
            answer_understander_dev = AnswerUnderstander(
                use_bert=use_bert,
                use_w2v=use_w2v,
                rnn_unit='lstm',
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=None,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=False,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                extra_feature_dim=extra_feature_dim,
                ans_max_len=ans_max_len,
                que_max_len=que_max_len,
                char_w2v_embedding_matrix_path=char_embedding_matrix_path,
                word_w2v_embedding_matrix_path=word_embedding_matrix_path)
        saver = tf.train.Saver()
        sess = tf.Session()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        saver.restore(sess, best_path)
        choose_acc = evaluation(sess, answer_understander_dev,
                                data_stream_choose_dev, 'result_{}.txt'.format(loop_index))
        print("the final choose accuracy:{}".format(choose_acc))
        return choose_acc


if __name__ == "__main__":
    with open('config_choose.json', encoding='utf-8') as infile:
        config = json.load(infile)

    int2bool = {1: True, 0: False}

    sentiment_words_path = config["train_config"]["SENTIMENT_WORDS_PATH"]
    batch_size = config["train_config"]["BATCH_SIZE"]
    is_loop = int2bool[config["train_config"]["Is_LOOP"]]
    is_sort = int2bool[config["train_config"]["IS_SORT"]]
    dropout_rate = config["train_config"]["DROPOUT_RATE"]
    nb_classes = config["train_config"]["NB_CLASSES"]
    attention_dim = config["train_config"]["ATTENTION_DIM"]
    nb_hops = config["train_config"]["NB_HOPS"]
    drop_template_path = config["train_config"]["DROP_CHOOSE_TEMPLATE_PATH"]
    use_bert = int2bool[config["train_config"]["USE_BERT"]]
    optimizer = config["train_config"]["OPTIMIZER"]
    learning_rate = config["train_config"]["LEARNING_RATE"]
    grad_clipper = config["train_config"]["GRAD_CLIPPER"]
    drop_choose_dev_path = config["train_config"]["DROP_CHOOSE_DEV_PATH"]
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
    ans_max_len = config["train_config"]["ANS_MAX_LEN"]
    que_max_len = config["train_config"]["QUE_MAX_LEN"]
    sentiment_polarity_multiple = config["train_config"]["POLARITY_MULTIPLE"]
    use_w2v = True
    if use_bert:
        use_w2v = False

    char_voc_path = config["w2v_config"]["CHAR_VOC_PATH"]
    char_embedding_matrix_path = config["w2v_config"][
        "CHAR_EMBEDDING_MATRIX_PATH"]
    word_voc_path = config["w2v_config"]["WORD_VOC_PATH"]
    word_embedding_matrix_path = config["w2v_config"][
        "WORD_EMBEDDING_MATRIX_PATH"]

    bert_model_path = config["bert_config"]["BERT_MODEL_PATH"]
    bert_config_file = config["bert_config"]["CONFIG_FILE"]
    bert_checkpoint_path = config["bert_config"]["INIT_CHECKPOINT"]
    bert_voc_path = config["bert_config"]["VOC_FILE"]
    sen2id_path = config["bert_config"]["SEN2ID_PATH"]
    question2targets = gather_targets_for_samples([], False,
                                                  question2targets_path)

    choose_samples, _, _ = read_file(drop_choose_dev_path)
    # choose_template, _, _ = read_file(drop_template_path)
    max_sequence_len = max(
        max([len(sample['question']) for sample in choose_samples]),
        max([len(sample['answer']) for sample in choose_samples]))
    with open(char_voc_path, 'rb') as infile:
        char_voc = pickle.load(infile)
    with open(word_voc_path, 'rb') as infile:
        word_voc = pickle.load(infile)
    bert_encoder = BertEncoder(model_root=bert_model_path,
                               bert_config_file=bert_config_file,
                               init_checkpoint=bert_checkpoint_path,
                               vocab_file=bert_voc_path,
                               max_sequence_len=max_sequence_len,
                               embedding_batch=3,
                               embedding_matrix_path=None,
                               sen2id_path=sen2id_path,
                               vec_dim=768)
    instances_choose_dev = make_instances(
        choose_samples,
        char_voc,
        word_voc,
        sentiment_words_path,
        ner_dict_path=ner_dict_path,
        pos_dict_path=pos_dict_path,
        use_extra_feature=use_extra_feature,
        question2targets=question2targets,
        is_training=False,
        need_augment=False)
    # instances_choose_template = make_instances(
    #     choose_template,
    #     char_voc,
    #     word_voc,
    #     sentiment_words_path,
    #     ner_dict_path=ner_dict_path,
    #     pos_dict_path=pos_dict_path,
    #     use_extra_feature=use_extra_feature,
    #     question2targets=question2targets,
    #     is_training=False,
    #     need_augment=False)

    # data_stream_choose_template = DataStream(instances=instances_choose_template,
    #                                          is_shuffle=False,
    #                                          is_loop=is_loop,
    #                                          batch_size=batch_size,
    #                                          ans_max_len=ans_max_len,
    #                                          que_max_len=que_max_len,
    #                                          use_bert=use_bert,
    #                                          bert_encoder=bert_encoder,
    #                                          is_sort=False)
    data_stream_choose_dev = DataStream(instances=instances_choose_dev,
                                        is_shuffle=False,
                                        is_loop=is_loop,
                                        batch_size=batch_size,
                                        ans_max_len=ans_max_len,
                                        que_max_len=que_max_len,
                                        use_bert=use_bert,
                                        bert_encoder=bert_encoder,
                                        is_sort=False)

    # get match answer
    # question2id, question2answer, question2label = \
    #     gather_question_template(choose_template, mode='choose')

    with tf.Graph().as_default():
        with tf.variable_scope("Model", reuse=None,
                               initializer=tf.glorot_uniform_initializer()):
            answer_understander_dev = AnswerUnderstander(
                use_bert=use_bert,
                use_w2v=use_w2v,
                rnn_unit='lstm',
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                learning_rate=learning_rate,
                grad_clipper=grad_clipper,
                global_step=None,
                attention_dim=attention_dim,
                nb_hops=nb_hops,
                rnn_dim=rnn_dim,
                lambda_l2=lambda_l2,
                is_training=False,
                sentiment_polarity_multiple=sentiment_polarity_multiple,
                nb_classes=nb_classes,
                use_extra_feature=use_extra_feature,
                ner_dict_size=ner_dict_size,
                pos_dict_size=pos_dict_size,
                extra_feature_dim=extra_feature_dim,
                ans_max_len=ans_max_len,
                que_max_len=que_max_len,
                char_w2v_embedding_matrix_path=char_embedding_matrix_path,
                word_w2v_embedding_matrix_path=word_embedding_matrix_path)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, best_path)

        # get the template representation and corresponding label
        # template_repr = []
        # template_label = []
        # for batch_index in range(data_stream_choose_template.get_nb_batch()):  # for each batch
        #     cur_batch = data_stream_choose_template.get_batch(batch_index)
        #     feed_dict = answer_understander_dev.create_feed_dict(cur_batch)
        #     template_repr.append(sess.run(answer_understander_dev.answer_repr,
        #                                   feed_dict=feed_dict))
        #     template_label.append(cur_batch.truths)
        # template_repr = np.concatenate(template_repr)
        # template_label = np.concatenate(template_label)
        # template_target = np.array([instance['target'] for instance in instances_choose_template])
        # template_answer = np.array([instance['answer'] for instance in instances_choose_template])

        # dev_repr = []
        # for batch_index in range(data_stream_choose_dev.get_nb_batch()):  # for each batch
        #     cur_batch = data_stream_choose_dev.get_batch(batch_index)
        #     feed_dict = answer_understander_dev.create_feed_dict(cur_batch)
        #     dev_repr.append(sess.run(answer_understander_dev.answer_repr,
        #                              feed_dict=feed_dict))
        # dev_repr = np.concatenate(dev_repr)

        # get the best matching answer
        # match_labels = []
        # max_cosine = []
        # match_answers = []
        # for instance_index in range(len(instances_choose_dev)):
        #     instance = instances_choose_dev[instance_index]
        #     target = instance['target']
        #     question = instance['question']
        #     question = '{}&&&&{}'.format(question, target)
        #     instance_template_repr = np.concatenate(
        #         [np.reshape(template_repr[id, :], [1, -1]) for id in question2id[question]], axis=-0)
        #     instance_ans_repr = dev_repr[instance_index, :]
        #     cos_distance = (instance_template_repr.dot(instance_ans_repr.T)) / (np.linalg.norm(instance_ans_repr) * (
        #         np.sqrt(np.sum(instance_template_repr * instance_template_repr, axis=-1))))
        #     match_labels.append(np.array(question2label[question])[np.argmax(cos_distance)])
        #     match_answers.append(template_answer[np.array(question2id[question])[np.argmax(cos_distance)]])
        #     max_cosine.append(cos_distance.max())

        # analysis the result
        # max_cosine = np.array(max_cosine)
        accuracy, predict_result, true_result = evaluation(sess, answer_understander_dev,
                                                           data_stream_choose_dev, 'result_{}.txt'.format(1))
        # match_indicator = np.array(match_labels) == np.array(true_result)
        # algorithm_indicator = np.array(predict_result) == np.array(true_result)
        # different_indicator = match_indicator != algorithm_indicator
        # valid_indicator = different_indicator & match_indicator
        # valid_cosine = max_cosine[valid_indicator]
        # true_cosine = max_cosine[match_indicator]
        # plt.hist(valid_cosine, bins=25, normed=True, alpha=0.5, histtype='stepfilled',
        #          color='steelblue', edgecolor='none')
        # plt.show()

        # match_accuracy = np.sum(np.array(match_labels) == np.array(true_result)) / (np.array(true_result).shape[0])
        # fused_result = [match_labels[i] if float(max_cosine[i]) > 0 else predict_result[i] for i in
        #                 range(len(predict_result))]
        # fused_accuracy = np.sum(np.array(fused_result) == np.array(true_result)) / (np.array(true_result).shape[0])
