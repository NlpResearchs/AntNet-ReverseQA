import numpy as np
import tensorflow as tf
import model_utils

from data_stream import InstanceBatch


class AnswerUnderstander(object):
    def __init__(self, use_bert, use_w2v, rnn_unit, dropout_rate,
                 char_w2v_embedding_matrix_path, rnn_dim, nb_classes,
                 optimizer, learning_rate, grad_clipper, global_step, nb_hops,
                 attention_dim, is_training, use_extra_feature, ans_max_len,
                 que_max_len, extra_feature_dim, ner_dict_size, pos_dict_size,
                 lambda_l2, sentiment_polarity_multiple,
                 word_w2v_embedding_matrix_path):
        self.use_bert = use_bert
        self.use_w2v = use_w2v
        self.rnn_unit = rnn_unit
        self.dropout_rate = dropout_rate
        self.rnn_dim = rnn_dim
        self.nb_classes = nb_classes
        self.is_training = is_training
        self.sentiment_polarity_multiple = sentiment_polarity_multiple
        self.nb_hops = nb_hops
        self.attention_dim = attention_dim
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_clipper = grad_clipper
        self.use_extra_feature = use_extra_feature
        self.ner_dict_size = ner_dict_size
        self.pos_dict_size = pos_dict_size
        self.extra_feature_dim = extra_feature_dim
        self.global_step = global_step
        self.lambda_l2 = lambda_l2
        self.ans_max_len = ans_max_len
        self.que_max_len = que_max_len
        assert self.use_w2v or self.use_bert

        # create placeholders
        self.que_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.ans_lens = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.truths = tf.placeholder(tf.int32, [None])  # [batch_size]

        self.que_skeleton_label = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_cw2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_ww2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_cw2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_ww2v_index_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        self.in_que_sentiment_polarity_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_sentiment_polarity_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_que_indicate_target_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]
        self.in_ans_indicate_target_matrix = tf.placeholder(
            tf.int32, [None, None])  # [batch_size, max_sequence_len]

        if self.use_extra_feature:
            self.in_que_pos_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_que_ner_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_pos_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]
            self.in_ans_ner_index_matrix = tf.placeholder(
                tf.int32, [None, None])  # [batch_size, max_sequence_len]

        if self.use_bert:
            self.que_bert_matrix = tf.placeholder(tf.float32,
                                                  [None, None, 768])
            self.ans_bert_matrix = tf.placeholder(tf.float32,
                                                  [None, None, 768])

        # init basic embedding matrix
        char_w2v_embedding_matrix, word_w2v_embedding_matrix = model_utils.load_variable_from_file(
            char_w2v_embedding_matrix_path, word_w2v_embedding_matrix_path)
        self.char_w2v_embedding_matrix = tf.convert_to_tensor(
            char_w2v_embedding_matrix)
        self.word_w2v_embedding_matrix = tf.convert_to_tensor(
            word_w2v_embedding_matrix)
        # if self.use_bert:
        #     self.bert_embedding_matrix = tf.convert_to_tensor(bert_embedding_matrix)

        # create model
        self.create_model_graph()

    def create_feed_dict(self, cur_batch):
        assert isinstance(cur_batch, InstanceBatch)
        feed_dict = {
            self.que_lens: cur_batch.que_lens,
            self.ans_lens: cur_batch.ans_lens,
            self.truths: cur_batch.truths,
            self.in_ans_cw2v_index_matrix: cur_batch.ans_cw2v_index_matrix,
            self.in_ans_ww2v_index_matrix: cur_batch.ans_ww2v_index_matrix,
            self.in_que_cw2v_index_matrix: cur_batch.que_cw2v_index_matrix,
            self.in_que_ww2v_index_matrix: cur_batch.que_ww2v_index_matrix,
            self.in_que_indicate_target_matrix:
                cur_batch.que_indicate_target_matrix,
            self.in_ans_indicate_target_matrix:
                cur_batch.ans_indicate_target_matrix,
            self.in_que_sentiment_polarity_matrix:
                cur_batch.que_sentiment_polarity_matrix,
            self.in_ans_sentiment_polarity_matrix:
                cur_batch.ans_sentiment_polarity_matrix,
            self.que_skeleton_label: cur_batch.que_skeleton_label_matrix
        }
        if self.use_bert:
            feed_dict.update({
                self.que_bert_matrix: cur_batch.que_bert_matrix,
                self.ans_bert_matrix: cur_batch.ans_bert_matrix
            })
        if self.use_extra_feature:
            feed_dict.update({
                self.in_que_pos_index_matrix:
                    cur_batch.que_pos_index_matrix,
                self.in_que_ner_index_matrix:
                    cur_batch.que_ner_index_matrix,
                self.in_ans_pos_index_matrix:
                    cur_batch.ans_pos_index_matrix,
                self.in_ans_ner_index_matrix:
                    cur_batch.ans_ner_index_matrix,
            })
        return feed_dict

    def create_model_graph(self):
        # truths = tf.get_variable(self.truths, name='truths')
        self.labels = model_utils.make_label(self.truths, self.nb_classes)
        que_in_features = []
        ans_in_features = []
        # feature_dim = 0

        # w2v embedding
        if self.use_w2v:
            que_char_w2v_features = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix,
                ids=self.in_que_cw2v_index_matrix)
            que_in_features.append(que_char_w2v_features)
            que_word_w2v_features = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix,
                ids=self.in_que_ww2v_index_matrix)
            que_in_features.append(que_word_w2v_features)
            ans_char_w2v_features = tf.nn.embedding_lookup(
                params=self.char_w2v_embedding_matrix,
                ids=self.in_ans_cw2v_index_matrix)
            ans_in_features.append(ans_char_w2v_features)
            ans_word_w2v_features = tf.nn.embedding_lookup(
                params=self.word_w2v_embedding_matrix,
                ids=self.in_ans_ww2v_index_matrix)
            ans_in_features.append(ans_word_w2v_features)

        # bert embedding
        if self.use_bert:
            que_in_features = [self.que_bert_matrix]
            ans_in_features = [self.ans_bert_matrix]
        # que_bert_features = tf.nn.embedding_lookup(params=self.bert_embedding_matrix, ids=self.que_ids)
        # ans_bert_features = tf.nn.embedding_lookup(params=self.bert_embedding_matrix, ids=self.ans_ids)

        # add extra features
        if self.use_extra_feature:
            indicate_ner_matrix = tf.get_variable(
                name='indicate_ner_embedding',
                shape=[self.ner_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)
            indicate_pos_matrix = tf.get_variable(
                name='indicate_pos_embedding',
                shape=[self.pos_dict_size, self.extra_feature_dim],
                trainable=True,
                dtype=tf.float32)
            que_indicate_ner_features = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_que_ner_index_matrix)
            ans_indicate_ner_features = tf.nn.embedding_lookup(
                params=indicate_ner_matrix, ids=self.in_ans_ner_index_matrix)
            que_indicate_pos_features = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_que_pos_index_matrix)
            ans_indicate_pos_features = tf.nn.embedding_lookup(
                params=indicate_pos_matrix, ids=self.in_ans_pos_index_matrix)
            que_in_features.append(que_indicate_ner_features)
            que_in_features.append(que_indicate_pos_features)
            ans_in_features.append(ans_indicate_ner_features)
            ans_in_features.append(ans_indicate_pos_features)

        # indicate-target vectors
        indicate_target_matrix = np.concatenate(
            [np.zeros([1, 30]), 0.3 * np.ones([1, 30])], axis=0)
        indicate_target_matrix = tf.Variable(indicate_target_matrix,
                                             trainable=True,
                                             name="indicate_target_embedding",
                                             dtype=tf.float32)
        que_indicate_target_features = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_que_indicate_target_matrix)
        ans_indicate_target_features = tf.nn.embedding_lookup(
            params=indicate_target_matrix,
            ids=self.in_ans_indicate_target_matrix)
        que_in_features.append(que_indicate_target_features)
        ans_in_features.append(ans_indicate_target_features)
        print('que_indicate_target_features shape:',
              que_indicate_target_features)

        # sentiment-polarity vectors
        # sentiment-polarity map ,keep sentiment's location, complete polarity flip model
        sentiment_polarity_matrix = np.concatenate(
            [np.identity(3) for i in range(self.sentiment_polarity_multiple)],
            axis=1)
        sentiment_polarity_matrix = tf.Variable(
            sentiment_polarity_matrix,
            name="sentiment_polarity_matrix",
            trainable=False,
            dtype=tf.float32)
        ans_sentiment_polarity_features = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_ans_sentiment_polarity_matrix)
        que_sentiment_polarity_features = tf.nn.embedding_lookup(
            params=sentiment_polarity_matrix,
            ids=self.in_que_sentiment_polarity_matrix)
        que_in_features.append(que_sentiment_polarity_features)
        ans_in_features.append(ans_sentiment_polarity_features)

        # [batch_size, question_len, dim]
        in_question_repr = tf.concat(axis=2, values=que_in_features)
        # [batch_size, question_len, dim]
        in_answer_repr = tf.concat(axis=2, values=ans_in_features)
        print("in_question_repr shape:", in_question_repr.shape)
        print("in_answer_repr shape:", in_answer_repr.shape)
        in_answer_repr = model_utils.dropout_layer(in_answer_repr,
                                                   self.dropout_rate,
                                                   self.is_training)
        in_question_repr = model_utils.dropout_layer(in_question_repr,
                                                     self.dropout_rate,
                                                     self.is_training)

        # TODO: complete skeleton information indicator
        indicate_skeleton_matrix = self.que_skeleton_label

        # basic encode using bi-lstm
        assert self.rnn_unit == 'lstm' or self.rnn_unit == 'gru'
        question_bi = model_utils.my_rnn_layer(input_reps=in_question_repr,
                                               rnn_dim=self.rnn_dim,
                                               rnn_unit=self.rnn_unit,
                                               input_lengths=self.que_lens,
                                               scope_name='basic_encode',
                                               is_training=self.is_training,
                                               reuse=False)
        answer_bi = model_utils.my_rnn_layer(input_reps=in_answer_repr,
                                             rnn_dim=self.rnn_dim,
                                             rnn_unit=self.rnn_unit,
                                             input_lengths=self.ans_lens,
                                             scope_name='basic_encode',
                                             is_training=self.is_training,
                                             reuse=True)

        answer_bi_last = model_utils.collect_final_step_of_lstm(
            answer_bi, self.ans_lens-1)
        self.answer_repr = answer_bi_last
        question_target_repr = model_utils.get_target_representation(
            question_bi, self.in_que_indicate_target_matrix)
        question_bi = model_utils.dropout_layer(question_bi, self.dropout_rate,
                                                self.is_training)
        question_bi_last = model_utils.collect_final_step_of_lstm(
            question_bi, self.que_lens)

        # flip_model for answer
        answer_bi = model_utils.sentiment_polarity_flip(
            answer_bi, ans_sentiment_polarity_features, question_target_repr,
            self.sentiment_polarity_multiple, self.attention_dim,
            "sentiment_polarity_flip")
        answer_bi = model_utils.dropout_layer(answer_bi, self.dropout_rate,
                                              self.is_training)

        # get skeleton representation for question
        # judge
        question_skeleton_repr = model_utils.generate_skeleton_representation(
            question_bi, indicate_skeleton_matrix, question_bi_last)
        question_skeleton_repr = model_utils.dropout_layer(
            question_skeleton_repr, self.dropout_rate, self.is_training)
        # chooose
        # question_skeleton_repr = question_target_repr

        # get semantic representation for question
        question_semantic_repr = model_utils.generate_semantic_representation(
            question_skeleton_repr, question_bi, self.que_lens,
            self.attention_dim)

        # question_semantic_repr = model_utils.dropout_layer(
        #     question_semantic_repr, self.dropout_rate, self.is_training)
        print('question_semantic_repr.shape:', question_semantic_repr.shape)
        print('question_skeleton_repr.shape:', question_skeleton_repr.shape)

        # get question-aware representation
        question_aware_repr = model_utils.get_aware_repr(
            answer_bi, question_skeleton_repr, question_semantic_repr,
            self.nb_hops, self.attention_dim, self.rnn_dim, self.ans_max_len,
            self.ans_lens, self.lambda_l2)
        question_aware_repr = model_utils.dropout_layer(
            question_aware_repr, self.dropout_rate, self.is_training)
        answer_aware_repr = model_utils.multi_hop_match(
            answer_bi_last, question_bi, self.nb_hops, self.rnn_dim,
            self.attention_dim, 'answer_aware', self.que_max_len, self.que_lens,
            self.lambda_l2)
        end_repr = tf.concat([question_aware_repr, answer_aware_repr],
                             axis=-1)

        logits = model_utils.full_connect_layer(end_repr,
                                                self.nb_classes,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training)

        self.loss = model_utils.compute_cross_entropy(logits, self.labels)
        self.probs = tf.nn.softmax(logits)
        correct = tf.nn.in_top_k(tf.to_float(logits), self.truths, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.argmax(self.probs, 1)

        tvars = tf.trainable_variables()
        self.tvars = tvars

        if self.lambda_l2 > 0.0:
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + self.lambda_l2 * l2_loss

        if self.is_training:
            trainable_variables = tf.trainable_variables()
            if self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
            grads = model_utils.compute_gradients(self.loss,
                                                  trainable_variables)
            # TODO: compute grad_clipper
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clipper)
            self.train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables), global_step=self.global_step)
