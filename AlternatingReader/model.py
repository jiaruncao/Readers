#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import  numpy as np
class Readers_Model(object):

    def __init__(self,batch_size,lr,max_q_len,max_f_len,num_doc,hidden_size,external_embed):

        self.init_op = tf.global_variables_initializer()

        self.batch_size = batch_size
        self.max_f_len = max_f_len      #fact的长度
        self.hidden_size = hidden_size
        self.max_q_len = max_q_len           #question的长度
        self.learning_rate = lr
        self.num_class = 4         #类别：四选一
        self.num_doc = num_doc      #文档数
        self.embedding_size =   hidden_size   #word embedding
        self.word_embeddings = tf.Variable(external_embed,trainable=True,dtype=tf.float32
                                               )

        self.y_placeholder = tf.placeholder(tf.int32,shape=[None])
        self.label_placeholder =  tf.one_hot(self.y_placeholder,self.num_class)
        self.fact_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,self.num_class, self.num_doc, self.max_f_len))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_class,self.max_q_len),name='question')
    '''
    def add_placeholder(self):
        self.label_placeholder =  tf.placeholder(tf.int32,shape=(self.batch_size,self.num_class))
        self.fact_placeholder = tf.placeholder(tf.int32,shape=(self.batch_size,self.num_doc,self.max_f_len))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_q_len))

        #self.question_len_placeholder = tf.placeholder(tf.int32,shape=(self.batch_size,))
    '''

    def fact_encoding_layer(self):

        #得到embedding tensor
        inputs = tf.nn.embedding_lookup(self.word_embeddings,self.fact_placeholder)

        #4维到3维
        inputs = tf.unstack(inputs,axis = 1)
        fact_vec_list =[]

        for i in range(self.num_class):
            state_list = []
            with tf.variable_scope("GRU_{}".format(i)):
                print ('^^^^^^^^^^^')
                gru_cell = tf.contrib.rnn.GRUCell(self.max_f_len)
                input = tf.unstack(inputs[i],axis = 1)
                for seq in input:
                    outputs,last_state = tf.nn.dynamic_rnn(
                        gru_cell,
                        seq,
                        dtype=np.float32,
                        )
                    state_list.append(last_state)
            fact_vec_list.append(state_list)
        self.fact = fact_vec_list
        fact_vec_list = tf.stack(fact_vec_list,0)
        fact_vec_list = tf.stack(fact_vec_list, 0)
        fact_vec_list = tf.reshape(fact_vec_list,[self.num_class,self.num_doc,self.batch_size,self.max_f_len])
        return  fact_vec_list


    def quesiton_encoding_layer(self):
        #tf.reset_default_graph()
        # 与输入一样，先embedding在GRU，将最后的状态作为输出
        input = tf.nn.embedding_lookup(self.word_embeddings, self.question_placeholder)
        input = tf.unstack(input,axis= 1)
        gru_cell = tf.contrib.rnn.GRUCell(self.max_q_len)
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
        return output_list,last_state_list



    def DNN_layer1(self, input_question,input_fact):
        '''

        :param input_question:
        :param input_fact:
        :return:
        '''
        list_result = []
        input_question = tf.reshape(input_question,[self.num_class,self.batch_size,self.max_q_len])
        input_fact = tf.unstack(input_fact,axis = 0)

        for i in range(self.num_class):

            input_fact[i] = tf.unstack(input_fact[i],axis = 0)

            list_out_layer = []
            for j in input_fact[i]:

                inputs = tf.concat([input_question[i],j],1)
                weight = {'h1':tf.Variable(tf.random_normal([self.max_q_len+self.max_f_len,self.hidden_size])),
                           'out':tf.Variable(tf.random_normal([self.hidden_size,self.max_q_len]))
                           }

                bias = {'b1':tf.Variable(tf.random_normal([self.hidden_size])),
                        'out':tf.Variable(tf.random_normal([self.max_q_len]))
                        }

                h_layer = tf.add(tf.matmul(inputs,weight['h1']),bias['b1'])

                out_layer = tf.sigmoid(tf.add(tf.matmul(h_layer,weight['out']),bias['out']))
                list_out_layer.append(out_layer)
            list_result.append(list_out_layer)
        self.DNN1 = list_result
        return list_result




    def DNN_layer2(self, input_question,input_fact):
        '''

        :param input_question:
        :param input_fact:
        :return:
        '''
        list_result = []
        input_question = tf.reshape(input_question,[self.num_class,self.batch_size,self.max_q_len])

        input_fact = tf.unstack(input_fact, axis=0)
        for i in range(self.num_class):
            # DNN layer
            input_fact[i] = tf.unstack(input_fact[i],axis = 0)
            list_out_layer = []

            for j in input_fact[i]:

                inputs = tf.concat([input_question[i],j],1)

                weight = {'h1':tf.Variable(tf.random_normal([self.max_q_len+self.max_f_len,self.hidden_size])),
                           'out':tf.Variable(tf.random_normal([self.hidden_size,self.max_q_len]))
                           }

                bias = {'b1':tf.Variable(tf.random_normal([self.hidden_size])),
                        'out':tf.Variable(tf.random_normal([self.max_q_len]))
                        }

                h_layer = tf.add(tf.matmul(inputs,weight['h1']),bias['b1'])

                out_layer = tf.sigmoid(tf.add(tf.matmul(h_layer,weight['out']),bias['out']))
                list_out_layer.append(out_layer)
            list_result.append(list_out_layer)
        return list_result

    def max_pooling1(self, input):
        input = tf.stack([x for x in input], axis=0)
        self.DNN2 = input
        input = tf.transpose(input,[2,0,1,3])
        output = tf.reduce_max(input_tensor=input,axis=2)

        output = tf.reshape(output, [self.batch_size, self.num_class, self.max_q_len])
        self.maxpool1 = output
        return output


    def max_pooling2(self, input):
        input = tf.stack([x for x in input], axis=0)
        print ('Preliminary input is %s' % input)
        self.DNN2 = input
        input = tf.transpose(input,[2,0,1,3])
        output = tf.reduce_max(input_tensor=input,axis=2)
        output = tf.reshape(output, [self.batch_size, self.num_class, self.max_q_len])
        self.maxpool2 = output
        return output

    def answering_layer(self,pool_out):
        softmax_w = tf.get_variable("softmax_w", [self.batch_size, self.max_q_len,1])
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
        encoder_f = self.fact_encoding_layer()
        DNN1 = self.DNN_layer1(encoder_q,encoder_f)
        maxpool1 = self.max_pooling1(DNN1)
        DNN2 = self.DNN_layer2(maxpool1,encoder_f)
        maxpool2 = self.max_pooling2(DNN2)
        answer = self.answering_layer(maxpool2)
        self.loss,self.acc = self.train_op(answer)
    '''
    def train_op(self):
        return self.train
    '''