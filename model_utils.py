import pickle
import time
import json
import sys
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import nn_ops, math_ops


def load_variable_from_file(char_w2v_embedding_matrix_path,
                            word_w2v_embedding_matrix_path):
    with open(char_w2v_embedding_matrix_path, 'rb') as infile:
        char_w2v_embedding_matrix = pickle.load(infile)
    with open(word_w2v_embedding_matrix_path, 'rb') as infile:
        word_w2v_embedding_matrix = pickle.load(infile)
    return char_w2v_embedding_matrix, word_w2v_embedding_matrix


def sentiment_polarity_flip(in_repr,
                            polarity_ori_repr,
                            question_target_repr,
                            sentiment_polarity_multiple,
                            attention_dim,
                            scope_name,
                            reuse=False):
    # in_repr: [batch_size, max_len, dim]
    # sentiment_ori_repr: [batch_size, max_len, 3]

    # it's fatal to remember that tf.get_variable's shape params need be specified  which is different from
    # other functions just like (tf.reshape, tf.tile)
    input_shape = tf.shape(in_repr)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    feature_dim = in_repr.get_shape().as_list()[-1]
    target_dim = question_target_repr.get_shape().as_list()[-1]
    in_repr = tf.reshape(in_repr, [-1, feature_dim])
    question_target_repr = tf.tile(tf.expand_dims(
        question_target_repr, axis=1), [1, seq_length, 1])
    question_target_repr = tf.reshape(question_target_repr, [-1, target_dim])

    with tf.variable_scope(scope_name, reuse=reuse):
        # define variable
        w1 = tf.get_variable("flip_w1",
                             shape=[1, attention_dim],
                             dtype=tf.float32)
        w2 = tf.get_variable("flip_w2",
                             shape=[1, attention_dim],
                             dtype=tf.float32)
        wqt1 = tf.get_variable("wqt_1",
                               shape=[target_dim, attention_dim],
                               dtype=tf.float32)
        wqt2 = tf.get_variable("wqt_2",
                               shape=[target_dim, attention_dim],
                               dtype=tf.float32)
        wv1 = tf.get_variable("wv_1",
                              shape=[feature_dim, attention_dim],
                              dtype=tf.float32)
        wv2 = tf.get_variable("wv_2",
                              shape=[feature_dim, attention_dim],
                              dtype=tf.float32)
        in_val_1 = tf.reshape(
            tf.tanh(
                tf.matmul(question_target_repr, wqt1) +
                tf.matmul(in_repr, wv1)),
            [batch_size, seq_length, attention_dim])
        in_val_2 = tf.reshape(
            tf.tanh(
                tf.matmul(question_target_repr, wqt2) +
                tf.matmul(in_repr, wv2)),
            [batch_size, seq_length, attention_dim])

        # change pram's shape
        w1 = tf.tile(w1, [seq_length, 1])
        w2 = tf.tile(w2, [seq_length, 1])

        w1 = tf.tile(tf.expand_dims(w1, axis=0), [batch_size, 1, 1])
        w2 = tf.tile(tf.expand_dims(w2, axis=0), [batch_size, 1, 1])

        # flip
        s1 = tf.reshape(tf.reduce_sum(tf.multiply(in_val_1, w1), axis=2),
                        [batch_size, seq_length, 1])
        p1 = tf.sigmoid(s1)
        p1_ = tf.ones_like(p1) - p1

        s2 = tf.reshape(tf.reduce_sum(tf.multiply(in_val_2, w2), axis=-1),
                        [batch_size, seq_length, 1])
        p2 = tf.sigmoid(s2)
        p2_ = tf.ones_like(p2) - p2

        flipped_polarity_repr = []
        flipped_polarity_repr.append(p1)
        flipped_polarity_repr.append(tf.multiply(p1_, p2))
        flipped_polarity_repr.append(tf.multiply(p1_, p2_))
        # expand sentiment's dim
        flipped_polarity_repr = tf.concat(flipped_polarity_repr *
                                          sentiment_polarity_multiple,
                                          axis=-1)
        in_repr = tf.reshape(in_repr, [batch_size, seq_length, feature_dim])
        return tf.concat([in_repr, flipped_polarity_repr], axis=2)


def my_rnn_layer(input_reps,
                 rnn_dim,
                 rnn_unit='lstm',
                 input_lengths=None,
                 scope_name=None,
                 reuse=False,
                 is_training=True,
                 dropout_rate=0.2,
                 use_cudnn=False):
    # input_reps = dropout_layer(input_reps,
    #                            dropout_rate,
    #                            is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        if use_cudnn:
            inputs = tf.transpose(input_reps, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                1,
                rnn_dim,
                direction="bidirectional",
                name="{}_cudnn_bi_lstm".format(scope_name),
                dropout=dropout_rate if is_training else 0)
            outputs, _ = lstm(inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
        else:
            if rnn_unit == 'lstm':
                context_rnn_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(
                    rnn_dim, forget_bias=1., state_is_tuple=True)
                context_rnn_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(
                    rnn_dim, forget_bias=1., state_is_tuple=True)
            else:
                context_rnn_cell_fw = tf.nn.rnn_cell.GRUCell(
                    rnn_dim, forget_bias=1., state_is_tuple=True)
                context_rnn_cell_bw = tf.nn.rnn_cell.GRUCell(
                    rnn_dim, forget_bias=1., state_is_tuple=True)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                context_rnn_cell_fw,
                context_rnn_cell_bw,
                input_reps,
                dtype=tf.float32,
                sequence_length=input_lengths)
            outputs = tf.concat(rnn_outputs, axis=2, name='lstm_output')
            outputs = dropout_layer(outputs,
                                    dropout_rate=dropout_rate,
                                    is_training=is_training)
    return outputs


def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passage_length, dim]
    # lengths: [batch_size]

    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    #
    indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation,
                          indices,
                          name='last-forward-lstm')
    return result  # [batch_size, dim]


def attention_layer(query_repr,
                    value_repr,
                    value_lengths,
                    attention_type,
                    attention_dim,
                    scope_name='attention',
                    is_training=True,
                    dropout_rate=0.2):
    # query_repr: [batch_size, query_dim]
    # value_repr: [batch_size, max_seq_length, value_dim]
    # value_lengths: [batch_size]
    input_shape = tf.shape(value_repr)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    query_dim = query_repr.get_shape().as_list()[-1]
    value_dim = value_repr.get_shape().as_list()[-1]

    with tf.variable_scope(scope_name):
        if attention_type == 'additive':
            # change vector's shape
            query_repr = tf.reshape(tf.tile(query_repr, [1, seq_length]),
                                    [-1, query_dim])
            value_repr = tf.reshape(value_repr, [-1, value_dim])

            wq = tf.get_variable("attention_wq",
                                 shape=[query_dim, attention_dim],
                                 dtype=tf.float32)
            wv = tf.get_variable("attention_wv",
                                 shape=[value_dim, attention_dim],
                                 dtype=tf.float32)
            b = tf.get_variable("attention_b", [attention_dim],
                                dtype=tf.float32)
            v = tf.get_variable("attention_v", [attention_dim, 1],
                                dtype=tf.float32)

            attention_value = tf.matmul(query_repr, wq) + tf.matmul(
                value_repr, wv)
            attention_value = nn_ops.bias_add(attention_value, b)
            attention_value = tf.tanh(attention_value)
            attention_value = tf.reshape(attention_value, [-1, attention_dim])
            attention_value = tf.matmul(attention_value, v)
            attention_value = tf.reshape(attention_value,
                                         [batch_size, seq_length])
            # attention_value: [batch_size, seq_length]
            attention_value = get_attention_score(attention_value,
                                                  value_lengths)

            value_repr = tf.reshape(value_repr,
                                    [batch_size, seq_length, value_dim])

        else:
            value_repr = tf.reshape(value_repr, [-1, value_dim])
            query_repr = tf.reshape(tf.tile(query_repr, [1, seq_length]),
                                    [-1, query_dim])
            wq = tf.get_variable("attention_wq",
                                 shape=[query_dim, attention_dim],
                                 dtype=tf.float32)
            wv = tf.get_variable("attention_wv",
                                 shape=[value_dim, attention_dim],
                                 dtype=tf.float32)
            q_temp = tf.tanh(tf.matmul(query_repr, wq))
            v_temp = tf.tanh(tf.matmul(value_repr, wv))
            attention_value = tf.multiply(q_temp, v_temp)
            attention_value = tf.reduce_sum(attention_value, axis=-1)
            attention_value = tf.reshape(attention_value,
                                         [batch_size, seq_length])
            attention_value = get_attention_score(attention_value,
                                                  value_lengths)
            value_repr = tf.reshape(value_repr,
                                    [batch_size, seq_length, value_dim])

        return weighted_sum(attention_value, value_repr)


def get_attention_score(attention_value, value_lengths):
    max_seq_length = tf.shape(attention_value)[1]
    batch_size = tf.shape(attention_value)[0]
    attention_iter = tf.TensorArray(tf.float32,
                                    1,
                                    dynamic_size=True,
                                    infer_shape=False)
    attention_iter = attention_iter.unstack(attention_value)
    sentence_lens_iter = tf.TensorArray(tf.int32,
                                        1,
                                        dynamic_size=True,
                                        infer_shape=False)
    sentence_lens_iter = sentence_lens_iter.unstack(value_lengths)
    end_attention_value = tf.TensorArray(size=batch_size, dtype=tf.float32)

    def body(i, end_attention_value):
        cur_length = sentence_lens_iter.read(i)
        cur_value = attention_iter.read(i)
        end_attention_value = end_attention_value.write(
            i,
            tf.concat([
                tf.nn.softmax(tf.slice(cur_value, [0], [cur_length])),
                tf.zeros([max_seq_length - cur_length])
            ],
                axis=-1))
        # end_attention_value = end_attention_value.write(
        # i, [
        #     tf.nn.softmax(tf.slice(cur_value, [0, 0 ], [1, cur_length])),
        #     tf.zeros([max_seq_length - cur_length])
        # ])
        return (i + 1, end_attention_value)

    def condition(i, end_attention_value):
        return i < batch_size

    _, end_attention_value = tf.while_loop(cond=condition,
                                           body=body,
                                           loop_vars=(0, end_attention_value))
    end_attention_value = end_attention_value.stack()
    return attention_value


def weighted_sum(attention_score, in_value):
    # in_value: [batch_size, seq_length, feature_dim]
    # attention_score: [batch_size, seq_length]
    batch_size = tf.shape(in_value)[0]
    feature_dim = in_value.get_shape().as_list()[-1]
    attention_score = tf.tile(tf.expand_dims(attention_score, axis=-1),
                              [1, 1, feature_dim])
    end_repr = tf.reduce_sum(tf.multiply(attention_score, in_value), axis=1)
    return tf.reshape(end_repr, [batch_size, feature_dim])


def is_same(vec_1, vec_2):
    temp = tf.to_int32(tf.equal(vec_1, vec_2))
    return tf.equal(tf.reduce_sum(temp), tf.reduce_sum(tf.ones_like(temp)))


def generate_skeleton_representation(in_value, indicate_matrix,
                                     alternative_vector):
    # in_value: [batch_size, seq_length, feature_dim]
    # indicate_matrix: [batch_size, seq_length]
    # alternative_vector: [batch_size, feature_dim]
    batch_size = tf.shape(in_value)[0]
    feature_dim = in_value.get_shape().as_list()[-1]
    indicate_matrix = tf.to_float(
        tf.tile(tf.expand_dims(indicate_matrix, axis=-1), [1, 1, feature_dim]))

    # 1: type -1:target 0:other
    indicate_matrix_type = tf.maximum(
        indicate_matrix, tf.zeros_like(indicate_matrix, dtype=tf.float32))
    indicate_matrix_target = -tf.minimum(
        indicate_matrix, tf.zeros_like(indicate_matrix, dtype=tf.float32))

    has_type_indicator = tf.reduce_sum(tf.reduce_sum(indicate_matrix_type,
                                                     axis=-1),
                                       axis=-1)
    has_target_indicator = tf.reduce_sum(tf.reduce_sum(indicate_matrix_target,
                                                       axis=-1),
                                         axis=-1)
    target_num = tf.reduce_sum(indicate_matrix_target[:, :, 0], axis=-1)
    type_num = tf.reduce_sum(indicate_matrix_type[:, :, 0], axis=-1)
    target_num = tf.reciprocal(
        tf.maximum(target_num, tf.ones_like(target_num, dtype=tf.float32)))
    type_num = tf.reciprocal(
        tf.maximum(type_num, tf.ones_like(type_num, dtype=tf.float32)))
    target_num = tf.tile(tf.expand_dims(target_num, axis=-1), [1, feature_dim])
    type_num = tf.tile(tf.expand_dims(type_num, axis=-1), [1, feature_dim])

    has_type_indicator = tf.minimum(
        has_type_indicator, tf.ones_like(has_type_indicator, dtype=tf.float32))
    has_type_indicator = tf.ones_like(has_type_indicator) - has_type_indicator
    has_target_indicator = tf.minimum(
        has_target_indicator,
        tf.ones_like(has_target_indicator, dtype=tf.float32))
    has_target_indicator = tf.ones_like(
        has_target_indicator) - has_target_indicator

    has_type_matrix = tf.tile(tf.reshape(has_type_indicator, [batch_size, 1]),
                              [1, feature_dim])
    has_target_matrix = tf.tile(
        tf.reshape(has_target_indicator, [batch_size, 1]), [1, feature_dim])
    alternative_type_repr = tf.multiply(has_type_matrix, alternative_vector)
    alternative_target_repr = tf.multiply(has_target_matrix,
                                          alternative_vector)

    type_repr = tf.reduce_sum(tf.multiply(in_value, indicate_matrix_type),
                              axis=1)
    target_repr = tf.reduce_sum(tf.multiply(in_value, indicate_matrix_target),
                                axis=1)
    type_repr = tf.reshape(type_repr, shape=[batch_size, feature_dim
                                             ]) + alternative_type_repr
    target_repr = tf.reshape(target_repr, shape=[batch_size, feature_dim
                                                 ]) + alternative_target_repr
    target_repr = tf.multiply(target_repr, target_num)
    type_repr = tf.multiply(type_repr, type_num)

    w_type = tf.get_variable("fuse_w1",
                             shape=[feature_dim, feature_dim],
                             dtype=tf.float32)
    w_target = tf.get_variable("fuse_w2",
                               shape=[feature_dim, feature_dim],
                               dtype=tf.float32)
    b = tf.get_variable("fuse_b", shape=[feature_dim], dtype=tf.float32)
    skeleton_representation = tf.nn.tanh(
        nn_ops.bias_add(
            tf.matmul(type_repr, w_type) + tf.matmul(target_repr, w_target),
            b))
    return skeleton_representation


def get_target_representation(in_value, indicate_matrix):
    # in_value: [batch_size, seq_length, feature_dim]
    # indicate_matrix: [batch_size, seq_length]
    feature_dim = in_value.get_shape().as_list()[-1]
    indicate_matrix = tf.to_float(
        tf.tile(tf.expand_dims(indicate_matrix, axis=-1), [1, 1, feature_dim]))
    target_num = tf.reduce_sum(indicate_matrix[:, :, 0], axis=-1)
    target_num = tf.tile(tf.expand_dims(target_num, axis=-1), [1, feature_dim])
    target_num = tf.reciprocal(
        tf.maximum(target_num, tf.ones_like(target_num, dtype=tf.float32)))
    target_repr = tf.reduce_sum(tf.multiply(in_value, indicate_matrix),
                                axis=1)
    target_repr = tf.multiply(target_repr, target_num)
    w_target = tf.get_variable("turn_w",
                               shape=[feature_dim, feature_dim],
                               dtype=tf.float32)
    b = tf.get_variable("turn_b", shape=[feature_dim], dtype=tf.float32)
    target_repr = tf.nn.tanh(nn_ops.bias_add(
        tf.matmul(target_repr, w_target), b))
    return target_repr


def compute_cross_entropy(logits, truth):
    # logits: [batch_size, nb_classes]
    logits = tf.to_float(logits)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=truth))


def make_label(truths, nb_classes):
    # truth: [batch_size]
    label = tf.one_hot(truths, nb_classes, 1, 0)
    return tf.to_float(label)


def full_connect_layer(in_val,
                       nb_classes,
                       dropout_rate,
                       is_training,
                       scope=None):
    # in_val: [batch_size, feature_dim]
    # input_shape = tf.shape(in_val)
    # batch_size = input_shape[0]
    feature_dim = in_val.get_shape().as_list()[-1]
    #     feat_dim = input_shape[2]
    with tf.variable_scope(scope or "full_connect"):
        dense1 = tf.layers.dense(inputs=in_val,
                                 units=int(feature_dim / 2),
                                 activation=tf.nn.tanh)
        dense1 = dropout_layer(dense1, dropout_rate, is_training)
        logits = tf.layers.dense(inputs=dense1,
                                 units=nb_classes,
                                 activation=None)
    return logits


def multi_hop_match(aware_repr, answer_repr, nb_hops, rnn_dim, attention_dim,
                    scope_name, ans_max_len, ans_lens, l2_reg):
    # aware_repr: [batch_size, feature_dim]
    # answer_repr: [batch_size, seq_length, answer_dim]
    # nb_hops: int
    # attention: int
    # rnn_dim: int
    with tf.variable_scope(scope_name):
        assert nb_hops > 0
        batch_size = batch_size = tf.shape(answer_repr)[0]
        aware_dim = aware_repr.get_shape().as_list()[-1]
        answer_dim = answer_repr.get_shape().as_list()[-1]

        # init memory
        ones_temp = tf.to_float(
            tf.reshape(tf.ones([batch_size, ans_max_len]),
                       [batch_size, ans_max_len, 1]))
        memories = tf.concat([answer_repr, ones_temp], axis=-1)

        attention_ws = tf.get_variable(
            name='W_al',
            shape=[nb_hops, 1, rnn_dim + answer_dim + aware_dim + 1],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        attention_bs = tf.get_variable(
            name='B_al',
            shape=[nb_hops, 1, ans_max_len],
            initializer=tf.zeros_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        gru_r = tf.get_variable(
            name='W_r',
            shape=[rnn_dim, answer_dim + 1],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        gru_z = tf.get_variable(
            name='W_z',
            shape=[rnn_dim, answer_dim + 1],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        gru_g = tf.get_variable(
            name='W_g',
            shape=[rnn_dim, rnn_dim],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        gru_x = tf.get_variable(
            name='W_x',
            shape=[rnn_dim, answer_dim + 1],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
            dtype=tf.float32)
        gru_r_update = tf.get_variable(
            name='U_r',
            shape=[rnn_dim, rnn_dim],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        gru_z_update = tf.get_variable(
            name='U_z',
            shape=[rnn_dim, rnn_dim],
            initializer=tf.orthogonal_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        e = tf.zeros([batch_size, rnn_dim])
        scores_list = []
        aware_repr = tf.tile(tf.expand_dims(aware_repr, 1),
                             [1, ans_max_len, 1])

        for h in range(nb_hops):
            memories_iter = tf.TensorArray(tf.float32,
                                           1,
                                           dynamic_size=True,
                                           infer_shape=False)
            memories_iter = memories_iter.unstack(memories)
            e_iter = tf.TensorArray(tf.float32,
                                    1,
                                    dynamic_size=True,
                                    infer_shape=False)
            e_iter = e_iter.unstack(e)
            aware_iter = tf.TensorArray(tf.float32,
                                        1,
                                        dynamic_size=True,
                                        infer_shape=False)
            aware_iter = aware_iter.unstack(aware_repr)
            sentence_lens_iter = tf.TensorArray(tf.int32,
                                                1,
                                                dynamic_size=True,
                                                infer_shape=False)
            sentence_lens_iter = sentence_lens_iter.unstack(ans_lens)
            newe = tf.TensorArray(size=batch_size, dtype=tf.float32)
            score = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, newe, score):
                a = memories_iter.read(i)
                olde = e_iter.read(i)
                b = tf.tile(tf.expand_dims(olde, 0), [ans_max_len, 1])
                c = aware_iter.read(i)
                g = tf.matmul(
                    attention_ws[h],
                    tf.transpose(tf.concat([a, b, c], 1),
                                 perm=[1, 0])) + attention_bs[h]
                l = math_ops.to_int32(sentence_lens_iter.read(i))
                score_temp = tf.concat([
                    tf.nn.softmax(tf.slice(g, [0, 0], [1, l])),
                    tf.zeros([1, ans_max_len - l])
                ], 1)
                # score_temp = tf.nn.softmax(g)
                score = score.write(i, score_temp)
                i_AL = tf.reshape(tf.matmul(score_temp, a), [-1, 1])
                olde = tf.reshape(olde, [-1, 1])
                r = tf.nn.sigmoid(
                    tf.matmul(gru_r, i_AL) + tf.matmul(gru_r_update, olde))
                z = tf.nn.sigmoid(
                    tf.matmul(gru_z, i_AL) + tf.matmul(gru_z_update, olde))
                e0 = tf.nn.tanh(
                    tf.matmul(gru_x, i_AL) +
                    tf.matmul(gru_g, tf.multiply(r, olde)))
                newe_temp = tf.multiply(1 - z, olde) + tf.multiply(z, e0)
                newe = newe.write(i, newe_temp)
                return (i + 1, newe, score)

            def condition(i, newe, score):
                return i < batch_size

            _, newe_final, score_final = tf.while_loop(cond=condition,
                                                       body=body,
                                                       loop_vars=(0, newe,
                                                                  score))
            e = tf.reshape(newe_final.stack(), [-1, rnn_dim])
            batch_score = tf.reshape(score_final.stack(), [-1, ans_max_len])
            scores_list.append(batch_score)
    return e


def get_aware_repr(answer_repr, skeleton_repr, semantic_repr, nb_hops,
                   attention_dim, rnn_dim, ans_max_len, ans_lens, l2_reg):
    skeleton_match_repr = multi_hop_match(skeleton_repr,
                                          answer_repr,
                                          nb_hops,
                                          rnn_dim,
                                          attention_dim,
                                          'skeleton_match',
                                          ans_max_len,
                                          ans_lens,
                                          l2_reg=l2_reg)
    semantic_match_repr = multi_hop_match(semantic_repr,
                                          answer_repr,
                                          nb_hops,
                                          rnn_dim,
                                          attention_dim,
                                          'semantic_match',
                                          ans_max_len,
                                          ans_lens,
                                          l2_reg=l2_reg)
    return tf.concat([skeleton_match_repr, semantic_match_repr], axis=-1)


def generate_semantic_representation(skeleton_repr, question_repr, que_lengths,
                                     attention_dim):
    return attention_layer(skeleton_repr, question_repr, que_lengths,
                           'additive', attention_dim,
                           'get_semantic_representation')


def compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


def evaluation(sess, valid_graph, dev_data_stream, out_path=None):
    if out_path is not None:
        outfile = open(out_path, 'w', encoding='utf-8')
    total = 0
    correct = 0
    predict_result = []
    true_result = []
    for batch_index in range(dev_data_stream.get_nb_batch()):  # for each batch
        cur_batch = dev_data_stream.get_batch(batch_index)
        total += cur_batch.batch_size
        if valid_graph.use_bert:
            cur_batch.update_bert_vector()
        feed_dict = valid_graph.create_feed_dict(cur_batch)
        [cur_correct, predictions
         ] = sess.run([valid_graph.eval_correct, valid_graph.predictions],
                      feed_dict=feed_dict)
        correct += cur_correct
        if out_path is not None:
            for i in range(cur_batch.batch_size):
                current_line = "question:{}\tanswer:{}\ttarget:{}\ttruth:{}\tprediction:{}\n".format(
                    str(cur_batch.questions[i]), str(cur_batch.answers[i]),
                    str(cur_batch.targets[i]), str(cur_batch.truths[i]),
                    str(predictions[i]))
                outfile.writelines(current_line)
                predict_result.append(predictions[i])
                true_result.append(cur_batch.truths[i])
    accuracy = correct / float(total) * 100
    if out_path is not None:
        outfile.writelines("the final accuracy:{}".format(accuracy))
        outfile.close()
    return accuracy, predict_result, true_result


def train(nb_epoch, sess, saver, data_stream_train,
          data_stream_valid, answer_understander_train,
          answer_understander_valid, best_path):
    best_accuracy = -1
    for epoch in range(nb_epoch):
        print('Train in epoch %d' % epoch)
        num_batch = data_stream_train.get_nb_batch()
        total_loss = 0
        # training
        data_stream_train.shuffle()
        for batch_index in tqdm(range(data_stream_train.get_nb_batch())):
            cur_batch = data_stream_train.get_batch(batch_index)
            if answer_understander_train.use_bert:
                cur_batch.update_bert_vector()
            feed_dict = answer_understander_train.create_feed_dict(cur_batch)
            loss, _, = sess.run([
                answer_understander_train.loss,
                answer_understander_train.train_op
            ],
                feed_dict=feed_dict)
            total_loss += loss
        start_time = time.time()
        acc, _, _ = evaluation(sess, answer_understander_valid, data_stream_valid)
        duration = time.time() - start_time
        print('Epoch %d: loss = %.4f (%.3f sec)' %
              (epoch, total_loss / num_batch, duration))
        print("Accuracy: %.2f" % acc)
        print('Evaluation time: %.3f sec' % (duration))
        if acc >= best_accuracy:
            best_accuracy = acc
            saver.save(sess, best_path)
            print('model has saved to %s!' % best_path)
