#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
#tf.reset_default_graph()
import  numpy as np
import  model
import time
import Helper
import time
BATCH_SIZE = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=  True


if __name__ == '__main__':

    emb_matrix, char2id, id2char = Helper.get_embedding('Readers_word2vec')
    data_list = Helper.read_json_file('/home/cjr/data/train.json')
    data_list_train = data_list[:int(len(data_list) * 0.7)]
    data_list_test = data_list[int(len(data_list) * 0.7):]
    doc_path = '/home/cjr/data/txt/'

    with tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope('model',reuse=None,initializer=initializer):
            m = model.Readers_Model(128,0.001,20,30,3,64,emb_matrix,True)
            m.input_()


        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(100):
 
            costs = 0.0
            iters = 0
            accuracy = 0.0
            read_data_time = 0
            train_time = 0
            test_time = 0
            print"Training Epoch: %d ..." % (i + 1)
            for batch in Helper.next_neural_reasoner_model_data_batch(data_list_train,char2id,128,3,30,20,doc_path):
                fact, question, y = batch
                time1 = time.time()
                loss, acc, _ = sess.run(
                    [m.loss, m.acc, m.train],
                    {m.question_placeholder: question, m.fact_placeholder: fact, m.y_placeholder: y,
                     m.dropout_placeholder: 0.1})
                time2=  time.time()
                train_time += (time2-time1)
                costs += loss
                accuracy += acc
                iters += 1
            costs /= iters
            accuracy /= iters
            print ('training spends %.1f s' %train_time)

            print"Epoch: %d  Train Cost: %f  " % (i + 1, costs)
            print ('Train Accuracy is %0.2f ' % accuracy)
            if (i+1) % 10 == 0:
                print "model saving..."
                model_saver.save(sess,'./model_path/'+'-%d'%(i+1))
            accuracy_test = 0.0
            iters_test= 0
            for batch_test in Helper.next_neural_reasoner_model_data_batch(data_list_test, char2id, 128, 3, 30, 20,
                                                                      doc_path):
                fact_test, question_test, y_test = batch_test

                acc_test = sess.run([m.acc],
                               {m.question_placeholder: question_test, m.fact_placeholder: fact_test,
                                m.y_placeholder: y_test,
                                m.dropout_placeholder: 0.1}
                               )
                #print (acc_test)
                accuracy_test += float(acc_test[0])
                iters_test += 1
            accuracy_test /= iters_test
            print ('Test Accuracy is %0.2f ' % accuracy_test)


