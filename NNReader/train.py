#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
#tf.reset_default_graph()
import  numpy as np
import  model
import time
import Helper

BATCH_SIZE = 128

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=  True

def data_iterator(batch_size,num_steps,max_num_steps,q_data,f_data,label_data):
    q_data = np.array(q_data, dtype=np.int32)
    f_data = np.array(f_data,dtype=np.int32)
    label_data = np.array(label_data,dtype=np.int32)
    if num_steps < max_num_steps:
        fact = f_data[num_steps*batch_size:(num_steps+1)*batch_size]
        question = q_data[num_steps*batch_size:(num_steps+1)*batch_size]
        label = label_data[num_steps*batch_size:(num_steps+1)*batch_size]

    return question,fact,label



def run_epoch(sess, m, q_data,f_data,label, eval_op):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    max_step = len(q_data)//m.batch_size

    for step in range(max_step):
        q,f,y = data_iterator(m.batch_size,step,max_step,q_data,f_data,label)

        loss, accuracy ,q,f,dnn1,dnn2,mp1,mp2,logits,label_ph,softmax,_= sess.run([m.loss, m.acc,m.question,m.fact,m.DNN1,m.DNN2,m.maxpool1,m.maxpool2,m.logits,m.label_placeholder,m.softmax,eval_op], {m.question_placeholder: q, m.fact_placeholder: f, m.label_placeholder: y,
                                             m.dropout_placeholder: 0.1})

        costs += loss
        iters += 1
    return (costs/iters),accuracy



if __name__ == '__main__':

    emb_matrix, char2id, id2char = Helper.get_embedding('Readers_word2vec')
    data_list = Helper.read_json_file('/home/cjr/data/train.json')
    data_list_train = data_list[:int(len(data_list) * 0.7)]
    data_list_test = data_list[int(len(data_list) * 0.7):]
    doc_path = '/home/cjr/data/txt/'

    with tf.Session(config=config) as sess:
        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.variable_scope('model',reuse=None,initializer=initializer):
            m = model.Readers_Model(128,0.001,20,30,3,64,emb_matrix)
            m.input_()

        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())

        for i in range(50):
            costs = 0.0
            iters = 0
            accuracy = 0.0

            print"Training Epoch: %d ..." % (i + 1)
            for batch in Helper.next_neural_reasoner_model_data_batch(data_list_train,char2id,128,3,30,20,doc_path):
                fact, question, y = batch

                loss, acc, q, f, dnn1, dnn2, mp1, mp2, logits,  softmax, _ = sess.run(
                    [m.loss, m.acc, m.question, m.fact, m.DNN1, m.DNN2, m.maxpool1, m.maxpool2, m.logits,
                     m.softmax, m.train],
                    {m.question_placeholder: question, m.fact_placeholder: fact, m.y_placeholder: y,
                     m.dropout_placeholder: 0.1})
                #print ('loss is %.4f ,acc is %.2f' % (loss, accuracy))
                costs += loss
                accuracy += acc
                iters += 1
            costs /= iters
            accuracy /= iters

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
                accuracy_test += float(acc_test[0])
                iters_test += 1
            accuracy_test /= iters_test
            print ('Test Accuracy is %0.2f ' % accuracy_test)


