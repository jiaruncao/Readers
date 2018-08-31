#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import  numpy as np
class Readers_Model(object):

    def __init__(self,batch_size,lr,max_q_len,max_f_len,num_doc,hidden_size,external_embed,flag):

        self.init_op = tf.global_variables_initializer()
        self.train_flag = flag
        self.batch_size = batch_size
        self.max_f_len = max_f_len      #fact的长度
        self.hidden_size = hidden_size
        self.max_q_len = max_q_len           #question的长度
        self.learning_rate = lr
        self.num_class = 4         #类别：四选一
        self.num_doc = num_doc      #文档数
        #self.vocabulary_size = vocub_size        #word embedding
        self.embedding_size =  100   #word embedding
        self.word_embeddings = tf.Variable(external_embed,trainable=True,dtype=tf.float32
                                               )
        self.y_placeholder = tf.placeholder(tf.int32,shape=[None])
        self.label_placeholder =  tf.one_hot(self.y_placeholder,self.num_class)
        #self.label_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,self.num_class))
        self.fact_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,self.num_class, self.num_doc, self.max_f_len))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_class,self.max_q_len),name='question')


    def fact_encoding_layer(self):

        #得到embedding tensor
        inputs = tf.nn.embedding_lookup(self.word_embeddings,self.fact_placeholder)

        #4维到3维
        inputs = tf.unstack(inputs,axis = 1)
        fact_state_list =[]
        fact_output_list = []
        for i in range(self.num_class):
            state_list = []
            output_list = []
            with tf.variable_scope("GRU_{}".format(i)):
                gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
                input = tf.unstack(inputs[i],axis = 1)
                for seq in input:


                    outputs,last_state = tf.nn.dynamic_rnn(
                        gru_cell,
                        seq,
                        dtype=np.float32,
                        )
                    state_list.append(last_state)
                    output_list.append(outputs)

                #drop out
                #fact_vec = tf.nn.dropout(fact_vec,self.dropout_placeholder)
            fact_state_list.append(state_list)
            fact_output_list.append(output_list)
        self.fact = fact_state_list

        fact_state_list = tf.stack(fact_state_list, 0)
        fact_output_list = tf.stack(fact_output_list,0)
        fact_output_list = tf.stack(fact_output_list,0)

        return  fact_state_list,fact_output_list


    def quesiton_encoding_layer(self):
        #tf.reset_default_graph()

        # 与输入一样，先embedding在GRU，将最后的状态作为输出
        input = tf.nn.embedding_lookup(self.word_embeddings, self.question_placeholder)
        input = tf.unstack(input,axis= 1)
        gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        init_state = gru_cell.zero_state(self.batch_size,tf.float32)
        output_list =[]
        last_state_list = []
        for i in input:
            output, last_state = tf.nn.dynamic_rnn(gru_cell,
                                        i,
                                        dtype=np.float32,

                                            )
            output_list.append(output)
            last_state_list.append(last_state)


        self.question = last_state_list
        last_state_list = tf.stack(last_state_list,0)
        output_list = tf.stack(output_list,0)
        return last_state_list,output_list

    def _create_rnn_cell(self, single=False):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=0.5)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(3)])
        return cell


    def Gated_att_layer(self,questions,facts):
        fact_list = tf.unstack(facts,axis=0)
        question_list = tf.unstack(questions,axis = 0)
        output_list = []
        state_list = []
        for i in range(self.num_class):
            fact_ = tf.unstack(fact_list[i],axis = 0)
            merge_f = tf.concat([x for x in fact_],axis = 1)
            merge_all = tf.concat([question_list[i],merge_f],axis = 1)
            #Helper
            if self.train_flag:
                helper = tf.contrib.seq2seq.TrainingHelper(merge_all,sequence_length = [self.max_f_len for _ in range(self.batch_size)])
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(merge_all,sequence_length = [self.max_f_len for _ in range(self.batch_size)] )

            rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            #Attention
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.hidden_size,merge_all)
            att_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell=rnn_cell,
                                           attention_mechanism=attention_mechanism,
                                           attention_layer_size=self.hidden_size,
                                           name='Attention_Wrapper')
            initial_state = att_wrapper.zero_state(self.batch_size, tf.float32)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(att_wrapper,self.hidden_size)

            #decoder
            with tf.variable_scope('decoder'):
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    out_cell,
                    helper,
                    initial_state,
                    )
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,swap_memory = True)

            output = tf.reshape(final_outputs[0],[self.batch_size,self.hidden_size,self.max_f_len])
            state = final_state[0]
            output_list.append(output)
            state_list.append(state)
        return output_list,state_list

    def self_matching_layer(self,inputs):
        output_list = []
        state_list = []
        for input in inputs:
            input = tf.transpose(input,[0,2,1])
            if self.train_flag:
                helper = tf.contrib.seq2seq.TrainingHelper(input,sequence_length = [self.max_f_len for _ in range(self.batch_size)])
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(input,sequence_length = [self.max_f_len for _ in range(self.batch_size)] )

            rnn_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            #Attention
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.hidden_size,input)
            att_wrapper = tf.contrib.seq2seq.AttentionWrapper(cell=rnn_cell,
                                           attention_mechanism=attention_mechanism,
                                           attention_layer_size=self.hidden_size,
                                           name='Attention_Wrapper')
            initial_state = att_wrapper.zero_state(self.batch_size, tf.float32)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(att_wrapper,self.hidden_size)

            #decoder
            with tf.variable_scope('decoder'):
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    out_cell,
                    helper,
                    initial_state,
                    )
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,swap_memory = True)

            output = tf.reshape(final_outputs[0],[self.batch_size,self.hidden_size,self.max_f_len])
            state = final_state[0]
            output_list.append(output)
            state_list.append(state)

        state_result = tf.stack([x for x in state_list],axis=0)
        state_result = tf.transpose(state_result,[1,0,2])
        return state_result




    def answering_layer(self,pool_out):
        print (pool_out)

        softmax_w = tf.get_variable("softmax_w", [self.batch_size, self.hidden_size,1])
        softmax_b = tf.get_variable('softmax_b',[self.batch_size,1,1])
        logits = tf.add((tf.matmul(pool_out, softmax_w)),softmax_b)

        logits=  tf.reshape(logits,[self.batch_size,self.num_class])
        return logits

    def train_op(self,logits):
        self.logits = logits

        with tf.name_scope('loss'):
            predict = tf.nn.softmax(logits)
            self.softmax = tf.nn.softmax(logits)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, tf.trainable_variables())
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.label_placeholder)+reg_term)
            cost = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope('train'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train = optimizer.apply_gradients(zip(grads, tvars))
            #self.train = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(predict,1),tf.argmax(self.label_placeholder,1))
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

        return  loss, accuracy

    def input_(self):

        _,encoder_q = self.quesiton_encoding_layer()
        _,encoder_f = self.fact_encoding_layer()
        gate_att,_ = self.Gated_att_layer(encoder_q,encoder_f)
        self_match = self.self_matching_layer(gate_att)
        answer = self.answering_layer(self_match)
        self.loss,self.acc = self.train_op(answer)
    '''
    def train_op(self):
        return self.train
    '''