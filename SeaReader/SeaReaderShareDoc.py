import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class SeaReaderShareDoc(object):
    def __init__(self, document_length, document_num, statement_length, hidden_dim, batch_size, answer_num,
                 emb_dimension, external_embedding_matrix, learning_rate,session, vacab_num=None):

        self.document_length = document_length
        self.document_num = document_num
        self.statement_length = statement_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.answer_num = answer_num
        self.emb_dimension = emb_dimension
        self._learning_rate = learning_rate

        # if there is not external emb
        self.vacab_num = vacab_num

        self._sess = session

        grad_norm_clip = 5.
        l2_reg_coef = 1e-4

        # init
        self._build_placeholders()
        self._build_variables(external_embedding_matrix)

        # Regularization
        tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_reg_coef), [self.embedding_matrix])

        # context layer
        self._context_layer()

        # dual path attention layer
        self._dual_path_attention_layer()

        # reasoning layer
        self._reasoning_layer()

        # integration & decision layer
        self._integration_decision_layer()

        # loss
        # convert label to one-hot
        self.one_hot_y = tf.one_hot(self.y, self.answer_num)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self._decision_forward_output, labels=self.one_hot_y))

        with tf.name_scope("optimizer"):
            self._learning_rate = tf.train.exponential_decay(self._learning_rate,
                                                       global_step=self._global_step,
                                                       decay_steps=100, decay_rate=0.9,staircase=True)

            self._opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            grads_and_vars = self._opt.compute_gradients(self.loss)
            capped_grads_and_vars = [(tf.clip_by_norm(g, grad_norm_clip), v) for g, v in grads_and_vars]
            self._train_op = self._opt.apply_gradients(capped_grads_and_vars, global_step=self._global_step)

        self._sess.run(tf.global_variables_initializer())

    def _context_layer(self):
        # embed
        self.statement_emb = tf.nn.embedding_lookup(self.embedding_matrix, self.statement)
        self.documents_emb = tf.nn.embedding_lookup(self.embedding_matrix, self.documents)

        # move the statement_length forward
        self.statement_emb = tf.transpose(self.statement_emb, [2, 0, 1, 3])
        self.statement_emb = tf.reshape(self.statement_emb, [-1, self.emb_dimension])
        self.statement_emb = tf.split(self.statement_emb, self.statement_length)

        self.documents_emb = tf.transpose(self.documents_emb, [2, 0, 1, 3])
        self.documents_emb = tf.reshape(self.documents_emb, [-1, self.emb_dimension])
        self.documents_emb = tf.split(self.documents_emb, self.document_length)

        # generate context presentation
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        statement_bilstm_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_cell_fw, lstm_cell_bw, self.statement_emb,
                                                                      dtype=tf.float32)
        documents_bilstm_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_cell_fw, lstm_cell_bw, self.documents_emb,
                                                                      dtype=tf.float32)

        statement_bilstm_outputs = self._transpose_outputs(statement_bilstm_outputs)
        documents_bilstm_outputs = self._transpose_outputs(documents_bilstm_outputs)

        self.statement_context = tf.reshape(statement_bilstm_outputs,
                                            [-1, self.answer_num * self.statement_length,
                                             2 * self.hidden_dim])

        self.documents_context = tf.reshape(documents_bilstm_outputs,
                                            [-1, self.document_num * self.document_length,
                                             2 * self.hidden_dim])

        # dropout
        self.statement_context = tf.nn.dropout(self.statement_context, self._keep_prob)
        self.documents_context = tf.nn.dropout(self.documents_context, self._keep_prob)

    def _transpose_outputs(self, bilstm_outputs):
        bilstm_outputs = tf.transpose(bilstm_outputs, [1, 0, 2])
        return bilstm_outputs

    def _dual_path_attention_layer(self):
        # matching matrix
        self.M_matrix = tf.matmul(self.statement_context, self.documents_context, transpose_b=True)

        # question-centric
        self.R_question_matrix = list()
        for i in range(0, self.document_num):
            alpha_m = tf.nn.softmax(tf.slice(self.M_matrix, [0, 0, i * self.document_length],
                                             [-1, -1, self.document_length]))
            self.R_question_matrix.append(tf.matmul(alpha_m, tf.slice(self.documents_context,
                                                                      [0, i * self.document_length, 0],
                                                                      [-1, self.document_length, -1])))
        self.R_question_matrix = tf.concat(self.R_question_matrix, -1)

        # document-centric
        self.R_document_matrix = list()
        for i in range(0, self.answer_num):
            alpha_document_m = tf.nn.softmax(
                tf.transpose(
                    tf.slice(self.M_matrix, [0, i * self.statement_length, 0], [-1, self.statement_length, -1]),
                    [0, 2, 1]))

            self.R_document_matrix.append(tf.matmul(alpha_document_m,
                                                    tf.slice(self.statement_context, [0, i * self.statement_length, 0],
                                                             [-1, self.statement_length, -1])))
        self.R_document_matrix = tf.concat(self.R_document_matrix, -1)

        # cross-document attention
        self.R_cross_matrix = []
        # for each statement
        for i in range(0, self.answer_num):
            self.concat_document_matric = tf.concat([self.documents_context,
                                                     tf.slice(self.R_document_matrix, [0, 0, i * 2 * self.hidden_dim],
                                                              [-1, -1, 2 * self.hidden_dim])], -1)
            self.M_cross_matrix = tf.matmul(self.concat_document_matric, self.concat_document_matric, transpose_b=True)
            self.beta_matrix = tf.nn.softmax(self.M_cross_matrix)
            self.R_cross_matrix.append(tf.matmul(self.beta_matrix, self.concat_document_matric))
        self.R_cross_matrix = tf.concat(self.R_cross_matrix, 1)

        # matching feature, mean pooling and max pooling
        self._extra_matching_feature()

    def _extra_matching_feature(self):
        # max pooling and mean pooling
        mean_pooling = tf.expand_dims(tf.reduce_mean(self.M_matrix, -1), -1)
        max_pooling = tf.expand_dims(tf.reduce_max(self.M_matrix, -1), -1)
        extra_feature_question = tf.concat(tf.concat([mean_pooling, max_pooling], -1), 2)
        self.R_question_matrix = tf.concat([self.R_question_matrix, extra_feature_question], -1)

        extra_feature_document = []
        for i in range(self.answer_num):
            slice_Matrix = tf.slice(self.M_matrix, [0, i * self.statement_length, 0], [-1, self.statement_length, -1])
            extra_feature_document.append(tf.concat(
                [tf.expand_dims(tf.reduce_mean(slice_Matrix, 1), -1),
                 tf.expand_dims(tf.reduce_max(slice_Matrix, 1), -1)], -1))

        self.R_cross_matrix = tf.concat([self.R_cross_matrix, tf.concat(extra_feature_document, 1)], -1)

    def _reasoning_layer(self):
        # gate layer
        # question 
        input_gate_question = tf.sigmoid(tf.matmul(tf.reshape(self.statement_context, [-1, 2 * self.hidden_dim]),
                                                   self._W_q_gate_reasoning_layer) + self._b_q_gate_reasoning_layer)
        input_gate_question = tf.reshape(input_gate_question, [-1, self.answer_num * self.statement_length, 1])

        # for gate loss
        self.input_gate_question = input_gate_question

        input_gate_question = tf.concat([input_gate_question] * (2 * self.document_num * self.hidden_dim + 2), -1)
        self._question_reasoning_after_gate = tf.multiply(self.R_question_matrix, input_gate_question)

        # document
        input_gate_document = tf.sigmoid(tf.matmul(tf.reshape(self.documents_context, [-1, 2 * self.hidden_dim]),
                                                   self._W_d_gate_reasoning_layer) + self._b_d_gate_reasoning_layer)
        input_gate_document = tf.reshape(input_gate_document,
                                         [-1, self.document_num * self.document_length, 1])

        # for gate loss
        self.input_gate_document = input_gate_document

        input_gate_document = tf.concat([input_gate_document] * (4 * self.hidden_dim + 2), -1)
        input_gate_document = tf.concat([input_gate_document] * self.answer_num, 1)
        self._document_reasoning_after_gate = tf.multiply(self.R_cross_matrix, input_gate_document)

        # LSTM
        # question
        self._question_reasoning_after_gate = tf.reshape(self._question_reasoning_after_gate,
                                                         [-1, self.answer_num,
                                                          self.statement_length,
                                                          2 * self.hidden_dim * self.document_num + 2])
        self._question_reasoning_after_gate = tf.transpose(self._question_reasoning_after_gate, [2, 0, 1, 3])
        self._question_reasoning_after_gate = tf.reshape(self._question_reasoning_after_gate,
                                                         [-1, self.document_num * self.hidden_dim * 2 + 2])
        self._question_reasoning_lstm_input = tf.split(self._question_reasoning_after_gate, self.statement_length)

        # document
        self._document_reasoning_after_gate = tf.reshape(self._document_reasoning_after_gate,
                                                         [-1, self.answer_num, self.document_num,
                                                          self.document_length, self.hidden_dim * 4 + 2])
        self._document_reasoning_after_gate = tf.transpose(self._document_reasoning_after_gate, [3, 0, 1, 2, 4])
        self._document_reasoning_after_gate = tf.reshape(self._document_reasoning_after_gate,
                                                         [-1, self.hidden_dim * 4 + 2])
        self._document_reasoning_lstm_input = tf.split(self._document_reasoning_after_gate, self.document_length)

        with tf.variable_scope('reasoning_layer_question'):
            # question
            reasoning_q_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            reasoning_q_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            reasoning_q_bilstm_outputs, _, _ = rnn.static_bidirectional_rnn(reasoning_q_lstm_cell_fw,
                                                                            reasoning_q_lstm_cell_bw,
                                                                            self._question_reasoning_lstm_input,
                                                                            dtype=tf.float32)
        with tf.variable_scope('reasoning_layer_document'):
            # document
            reasoning_d_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            reasoning_d_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            reasoning_d_bilstm_outputs, _, _ = rnn.static_bidirectional_rnn(reasoning_d_lstm_cell_fw,
                                                                            reasoning_d_lstm_cell_bw,
                                                                            self._document_reasoning_lstm_input,
                                                                            dtype=tf.float32)

        # process output
        reasoning_q_bilstm_outputs = self._transpose_outputs(reasoning_q_bilstm_outputs)
        reasoning_d_bilstm_outputs = self._transpose_outputs(reasoning_d_bilstm_outputs)

        reasoning_q_bilstm_outputs = tf.reshape(reasoning_q_bilstm_outputs,
                                                [-1, self.answer_num,
                                                 self.statement_length,
                                                 2 * self.hidden_dim])
        reasoning_d_bilstm_outputs = tf.reshape(reasoning_d_bilstm_outputs,
                                                [-1, self.answer_num, self.document_num,
                                                 self.document_length,
                                                 2 * self.hidden_dim])

        max_pooling_q = tf.reduce_max(reasoning_q_bilstm_outputs, 2)
        max_pooling_d = tf.reduce_max(reasoning_d_bilstm_outputs, 3)

        self._reasoning_layer_output = (max_pooling_q, max_pooling_d)

    def _integration_decision_layer(self):

        reasoning_layer_q, reasoning_layer_d = self._reasoning_layer_output

        # gate layer
        input_gate = tf.sigmoid(tf.matmul(tf.reshape(reasoning_layer_d, [-1, 2 * self.hidden_dim]),
                                          self._W_gate_decision_layer) + self._b_gate_decision_layer)
        input_gate_ = tf.reshape(input_gate, [-1, self.answer_num, self.document_num])
        input_gate_ = tf.expand_dims(input_gate_, -1)
        input_gate_ = tf.concat([input_gate_] * (2 * self.hidden_dim), -1)
        self._decision_after_gate = tf.multiply(reasoning_layer_d, input_gate_)

        # pooling
        d_max = tf.reduce_max(self._decision_after_gate, -2)
        d_mean = tf.reduce_mean(self._decision_after_gate, -2)
        self._decision_pooling_output = tf.concat([d_max, d_mean], -1)

        feed_forward_input = tf.concat([reasoning_layer_q, self._decision_pooling_output], -1)

        # feed forward
        self._decision_forward_output = tf.matmul(tf.reshape(feed_forward_input, [-1, 6 * self.hidden_dim]),
                                                  self._W_feed_forward) + self._b_feed_forward
        self._decision_forward_output = tf.reshape(self._decision_forward_output, [-1, self.answer_num])

    def _build_variables(self, external_embedding_matrix):

        if self.vacab_num is None:
            # when we have pretrained emb, use this code
            self.embedding_matrix = tf.Variable(external_embedding_matrix, trainable=True, name="emb", dtype=tf.float32)

        with tf.variable_scope("variables",
                               initializer=tf.random_normal_initializer(mean=0.0, stddev=0.22, dtype=tf.float32)):
            if self.vacab_num is not None:
                # randomly initialize emb
                self.embedding = tf.get_variable("emb", [self.vacab_num, self.emb_dimension], trainable=True,
                                                 dtype=tf.float32)

            # gate in reasoning layer
            self._W_q_gate_reasoning_layer = tf.get_variable("W_q_gate_r_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_q_gate_reasoning_layer = tf.get_variable("b_q_gate_r_l", [], dtype=tf.float32)

            self._W_d_gate_reasoning_layer = tf.get_variable("W_d_gate_r_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_d_gate_reasoning_layer = tf.get_variable("b_d_gate_r_l", [], dtype=tf.float32)

            # gate in decision layer
            self._W_gate_decision_layer = tf.get_variable("W_gate_d_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_gate_decision_layer = tf.get_variable("b_gate_d_l", [], dtype=tf.float32)

            # feed forward network
            self._W_feed_forward = tf.get_variable("W_forward", [6 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_feed_forward = tf.get_variable("b_forward", [], dtype=tf.float32)

            self._global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                                dtype=tf.int32, trainable=False)

    def _build_placeholders(self):
        # question-answer pair
        self.statement = tf.placeholder(tf.int32, [None, self.answer_num, self.statement_length])
        # documents
        self.documents = tf.placeholder(tf.int32,
                                        [None, self.document_num, self.document_length])
        # answer
        # e.g. 2 if the answer is C
        self.y = tf.placeholder(tf.int32, [None])

        self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    def batch_fit(self, documents, statements, answers):
        """
        Perform a batch training iteration
        """
        feed_dict = {
            self.documents: documents,
            self.statement: statements,
            self.y: answers,
            self._keep_prob: 0.8,
            # self._learning_rate: learning_rate
        }

        loss, _, step, prediction = self._sess.run(
            [self.loss, self._train_op, self._global_step, self._decision_forward_output],
            feed_dict=feed_dict)
        return loss, step, prediction

    def batch_predict(self, documents, statements, answers):
        """
        Perform batch prediction. Computes accuracy of batch predictions.
        """
        feed_dict = {
            self.documents: documents,
            self.statement: statements,
            self.y: answers,
            self._keep_prob: 1.,
            # self._learning_rate: 0.
        }
        loss, attentions = self._sess.run(
            [self.loss, self._decision_forward_output],
            feed_dict=feed_dict)

        return loss, attentions




if __name__ == '__main__':
    SeaReader(2, 3, 4, 5, 6, 7, 8, 9, 10)
    # document_length, document_num, statement_length, hidden_dim, batch_size, answer_num,emb_dimension, embedding_matrix, vacab_num
