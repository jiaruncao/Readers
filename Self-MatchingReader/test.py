#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import  numpy as np
import Helper
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
import model



if __name__ == '__main__':
    emb_matrix, char2id, id2char = Helper.get_embedding('Readers_word2vec')
    data_list = Helper.read_json_file('/home/cjr/data/test.json')
    doc_path = '/home/cjr/data/txt/'

    #tf.reset_default_graph()

    with tf.Graph().as_default(),tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(-0.1,0.1)

        with tf.variable_scope('model',reuse=None,initializer=initializer):
            m = model.Readers_Model(128,0.001,20,30,3,64,emb_matrix
                                    ,False)
            m.input_()

            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            #saver = tf.train.Saver('./model_path/-50.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./model_path/'))

            costs = 0.0
            accuracy = 0.0
            iters = 0
            for batch in Helper.next_neural_reasoner_model_data_batch(data_list,char2id,128,3,30,20,doc_path):
                fact, question, y = batch
                loss, acc= sess.run(
                    [m.loss, m.acc],
                    {m.question_placeholder: question, m.fact_placeholder: fact, m.y_placeholder: y,
                     m.dropout_placeholder: 0.1})
                # print ('loss is %.4f ,acc is %.2f' % (loss, accuracy))
                costs += loss
                accuracy += acc
                iters += 1
            costs /= iters
            accuracy /= iters
 
            print ('Test Acc is %.2f ' % accuracy)
