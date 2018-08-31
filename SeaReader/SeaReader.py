import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class SeaReader(object):
    def __init__(self, document_length, document_num, statement_length, hidden_dim, answer_num,
                 emb_dimension, external_embedding_matrix, learning_rate,session, vacab_num=None):

        self.document_length = document_length
        self.document_num = document_num
        self.statement_length = statement_length
        self.hidden_dim = hidden_dim
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

        # # Regularization
        # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_reg_coef), [self.embedding_matrix])

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
                                                       decay_steps=400, decay_rate=0.9,staircase=True)

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

        self.documents_emb = tf.transpose(self.documents_emb, [3, 0, 1, 2,4])
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
                                            [-1, self.answer_num , self.statement_length,
                                             2 * self.hidden_dim])

        self.documents_context = tf.reshape(documents_bilstm_outputs,
                                            [-1, self.answer_num,self.document_num , self.document_length,
                                             2 * self.hidden_dim])

        # dropout
        self.statement_context = tf.nn.dropout(self.statement_context, self._keep_prob)
        self.documents_context = tf.nn.dropout(self.documents_context, self._keep_prob)

    def _transpose_outputs(self, bilstm_outputs):
        bilstm_outputs = tf.transpose(bilstm_outputs, [1, 0, 2])
        return bilstm_outputs

    def _dual_path_attention_layer(self):
        # split docs
        documents_context_split = tf.split(self.documents_context, self.document_num,axis=-3)

        # M matrix
        self.M_matrix = []
        statement_context_expand = tf.expand_dims(self.statement_context,-3)
        for doc in documents_context_split:
            # matching matrix
            self.M_matrix.append(tf.matmul(statement_context_expand, doc, transpose_b=True))
        self.M_matrix_concat = tf.concat(self.M_matrix, -3)

        # question-centric
        alpha_m = tf.nn.softmax(self.M_matrix_concat)
        self.R_question_matrix = tf.matmul(alpha_m, self.documents_context)

        # document-centric
        self.R_document_matrix = []
        alpha_document_m = tf.nn.softmax(tf.transpose(self.M_matrix_concat,[0,1,2,4,3]))
        alpha_document_m = tf.split(alpha_document_m,self.document_num,-3)
        for am in alpha_document_m:
            self.R_document_matrix.append(tf.matmul(am,statement_context_expand))
        self.R_document_matrix = tf.concat(self.R_document_matrix, -3)

        # cross-document attention
        self.R_cross_matrix = []
        self.concat_document_matric = tf.concat([self.documents_context,self.R_document_matrix], -1)
        self.M_cross_matrix = tf.matmul(self.concat_document_matric, self.concat_document_matric, transpose_b=True)
        beta_matrix = tf.nn.softmax(self.M_cross_matrix)
        self.R_cross_matrix = tf.matmul(beta_matrix, self.concat_document_matric)

        # matching feature, mean pooling and max pooling
        self._extra_matching_feature()

    def _extra_matching_feature(self):
        # max pooling and mean pooling
        self.R_question_matrix = self._mean_and_max_pooling(-1,self.R_question_matrix)
        self.R_cross_matrix = self._mean_and_max_pooling(-2, self.R_cross_matrix)

    def _mean_and_max_pooling(self,axis,R_matrix):
        mean_pooling = tf.expand_dims(tf.reduce_mean(self.M_matrix_concat, axis), -1)
        max_pooling = tf.expand_dims(tf.reduce_max(self.M_matrix_concat, axis), -1)
        extra_feature_question = tf.concat([mean_pooling, max_pooling], -1)
        R_matrix = tf.concat([R_matrix, extra_feature_question], -1)

        return R_matrix

    def _reasoning_layer(self):
        # gate layer
        # question 
        input_gate_question = tf.sigmoid(tf.matmul(tf.reshape(self.statement_context, [-1, 2 * self.hidden_dim]),
                                                   self._W_q_gate_reasoning_layer) + self._b_q_gate_reasoning_layer)
        input_gate_question = tf.reshape(input_gate_question, [-1, self.answer_num, self.statement_length, 1])

        # TODO for gate loss
        # self.input_gate_question = input_gate_question
        input_gate_question=tf.expand_dims(input_gate_question,-3)
        input_gate_question = tf.concat([input_gate_question] * (self.document_num), -3)
        input_gate_question = tf.concat([input_gate_question] * (self.hidden_dim*2 +2), -1)
        self._question_reasoning_after_gate = tf.multiply(self.R_question_matrix, input_gate_question)

        # document
        input_gate_document = tf.sigmoid(tf.matmul(tf.reshape(self.documents_context, [-1, 2 * self.hidden_dim]),
                                                   self._W_d_gate_reasoning_layer) + self._b_d_gate_reasoning_layer)
        input_gate_document = tf.reshape(input_gate_document,
                                         [-1, self.answer_num,self.document_num,self.document_length, 1])

        # for gate loss
        # self.input_gate_document = input_gate_document

        input_gate_document = tf.concat([input_gate_document] * (4 * self.hidden_dim + 2), -1)
        self._document_reasoning_after_gate = tf.multiply(self.R_cross_matrix, input_gate_document)

        # LSTM
        # question
        # self._question_reasoning_after_gate = tf.reshape(self._question_reasoning_after_gate,
        #                                                  [-1, self.answer_num,
        #                                                   self.statement_length,
        #                                                   2 * self.hidden_dim * self.document_num + 2])
        self._question_reasoning_after_gate = tf.transpose(self._question_reasoning_after_gate, [3, 0, 1,2,4])
        self._question_reasoning_after_gate = tf.reshape(self._question_reasoning_after_gate,
                                                         [-1, self.hidden_dim * 2 + 2])
        self._question_reasoning_lstm_input = tf.split(self._question_reasoning_after_gate, self.statement_length)

        # document
        # self._document_reasoning_after_gate = tf.reshape(self._document_reasoning_after_gate,
        #                                                  [-1, self.answer_num, self.document_num,
        #                                                   self.document_length, self.hidden_dim * 4 + 2])
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
                                                [-1, self.answer_num,self.document_num,
                                                 self.statement_length,
                                                 2 * self.hidden_dim])
        reasoning_d_bilstm_outputs = tf.reshape(reasoning_d_bilstm_outputs,
                                                [-1, self.answer_num, self.document_num,
                                                 self.document_length,
                                                 2 * self.hidden_dim])

        max_pooling_q = tf.reduce_max(reasoning_q_bilstm_outputs, 3)
        max_pooling_d = tf.reduce_max(reasoning_d_bilstm_outputs, 3)

        self._reasoning_layer_output = (max_pooling_q, max_pooling_d)

    def _integration_decision_layer(self):

        reasoning_layer_q, reasoning_layer_d = self._reasoning_layer_output

        # gate layer
        input_gate_d = tf.sigmoid(tf.matmul(tf.reshape(reasoning_layer_d, [-1, 2 * self.hidden_dim]),
                                          self._W_gate_decision_layer_d) + self._b_gate_decision_layer_d)
        input_gate_d = tf.reshape(input_gate_d, [-1, self.answer_num, self.document_num])
        input_gate_d = tf.expand_dims(input_gate_d, -1)
        input_gate_d = tf.concat([input_gate_d] * (2 * self.hidden_dim), -1)
        self._decision_after_gate_d = tf.multiply(reasoning_layer_d, input_gate_d)

        # gate layer
        input_gate_q = tf.sigmoid(tf.matmul(tf.reshape(reasoning_layer_q, [-1, 2 * self.hidden_dim]),
                                          self._W_gate_decision_layer_q) + self._b_gate_decision_layer_q)
        input_gate_q_ = tf.reshape(input_gate_q, [-1, self.answer_num, self.document_num])
        input_gate_q_ = tf.expand_dims(input_gate_q_, -1)
        input_gate_q_ = tf.concat([input_gate_q_] * (2 * self.hidden_dim), -1)
        self._decision_after_gate_q = tf.multiply(reasoning_layer_q, input_gate_q_)

        # pooling
        d_max = tf.reduce_max(self._decision_after_gate_d, -2)
        d_mean = tf.reduce_mean(self._decision_after_gate_d, -2)
        self._decision_pooling_output_d = tf.concat([d_max, d_mean], -1)

        # pooling
        q_max = tf.reduce_max(self._decision_after_gate_q, -2)
        q_mean = tf.reduce_mean(self._decision_after_gate_q, -2)
        self._decision_pooling_output_q = tf.concat([q_max, q_mean], -1)

        feed_forward_input = tf.concat([self._decision_pooling_output_q, self._decision_pooling_output_d], -1)

        # feed forward
        self._decision_forward_output = tf.matmul(tf.reshape(feed_forward_input, [-1, 8 * self.hidden_dim]),
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
                self.embedding_matrix = tf.get_variable("emb", [self.vacab_num, self.emb_dimension], trainable=True,
                                                 dtype=tf.float32)

            # gate in reasoning layer
            self._W_q_gate_reasoning_layer = tf.get_variable("W_q_gate_r_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_q_gate_reasoning_layer = tf.get_variable("b_q_gate_r_l", [], dtype=tf.float32)

            self._W_d_gate_reasoning_layer = tf.get_variable("W_d_gate_r_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_d_gate_reasoning_layer = tf.get_variable("b_d_gate_r_l", [], dtype=tf.float32)

            # gate in decision layer
            self._W_gate_decision_layer_d = tf.get_variable("W_gate_d_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_gate_decision_layer_d = tf.get_variable("b_gate_d_l", [], dtype=tf.float32)

            self._W_gate_decision_layer_q = tf.get_variable("W_gate_q_l", [2 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_gate_decision_layer_q = tf.get_variable("b_gate_q_l", [], dtype=tf.float32)

            # feed forward network
            self._W_feed_forward = tf.get_variable("W_forward", [8 * self.hidden_dim, 1], dtype=tf.float32)
            self._b_feed_forward = tf.get_variable("b_forward", [], dtype=tf.float32)

            self._global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                                dtype=tf.int32, trainable=False)

    def _build_placeholders(self):
        # question-answer pair
        self.statement = tf.placeholder(tf.int32, [None, self.answer_num, self.statement_length])
        # documents
        self.documents = tf.placeholder(tf.int32,
                                        [None, self.answer_num, self.document_num, self.document_length])
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
        }
        loss, prediction = self._sess.run(
            [self.loss, self._decision_forward_output],
            feed_dict=feed_dict)

        return loss, prediction




if __name__ == '__main__':
    SeaReader(2, 3, 4, 5, 6, 7, 8, None,1e-4,None,vacab_num=10)
    # document_length, document_num, statement_length, hidden_dim, batch_size, answer_num,emb_dimension, embedding_matrix, vacab_num
