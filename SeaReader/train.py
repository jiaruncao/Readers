import os
import numpy as np
import sys
from pathlib import Path

up_two_level = str(Path(__file__).parents[2])
sys.path.append(up_two_level)
from Helper import Helper

from tqdm import tqdm


def run(config, sess, model, train_data, test_data, char2id, saver=None):
    # split for evaluate set
    evaluate_data = train_data[:config.evaluate_num]
    train_data = train_data[config.evaluate_num:]



    train_batch_num = int(np.math.ceil(len(train_data)*1.0 / config.batch_size))

    valid_acc = 0.4

    for epoch in range(config.num_epochs):
        train_batches = Helper.next_seareader_model_data_batch(config, train_data, char2id, config.batch_size)
        for batch in train_batches:

            X_train, Q_train, Y_train = batch
            batch_loss, step, attentions = model.batch_fit(X_train, Q_train, Y_train)

            iteration = step - epoch * train_batch_num
            # run valid set and save model
            if iteration % config.checkpoint_every == 0:
                print "current epoch:", epoch
                print "current batch:",iteration

                t_acc = compute_accuracy(attentions, Y_train)
                print "Train        loss: %.5f      ACC: %.5f" % (batch_loss, t_acc)

                loss, v_acc = run_epoch(config,model, evaluate_data, char2id, config.batch_size)
                print "Valid        loss: %.5f      ACC: % .5f" % (loss, v_acc)

                if v_acc > valid_acc:
                    ckpt_file = 'model-l{:.3f}_a{:.3f}.ckpt'.format(loss, v_acc)
                    path = saver.save(sess, os.path.join(config.ckpt_dir, ckpt_file), global_step=step)
                    valid_acc=v_acc
                    print('Saved model')

                    _, test_acc = run_epoch(config,model, test_data, char2id, config.batch_size)
                    print "Test     ACC: % .5f" % (test_acc)


def debug_run(config, sess, model, train_data, test_data, char2id, saver=None):
    train_data = train_data[:config.debug_train_num]
    # split for evaluate set
    # evaluate_data = train_data[:config.debug_evaluate_num]
    train_data = train_data[config.debug_evaluate_num:]

    train_batch_num = int(np.math.ceil(len(train_data)*1.0 / config.debug_batch_size))

    # batch_num = len(evaluate_data) / config.debug_batch_size


    for epoch in range(config.num_epochs):
        train_batches = Helper.next_seareader_model_data_batch(config, train_data, char2id, config.debug_batch_size)
        for batch in train_batches:

            X_train, Q_train, Y_train = batch
            batch_loss, step, attentions = model.batch_fit(X_train, Q_train, Y_train)
            # print step
            iteration = step - epoch * train_batch_num

            # run valid set and save model
            if iteration % config.debug_checkpoint_every == 0:
                print "current epoch:", epoch
                print "current batch:", iteration

                t_acc = compute_accuracy(attentions, Y_train)
                print "Train        loss: %.5f      ACC: %.5f" % (batch_loss, t_acc)

                # loss, v_acc = run_epoch(model, evaluate_set, batch_num)
                # print "Valid        loss: %.5f      ACC: % .5f" % (loss, v_acc)
        # print 'ok'

def compute_accuracy(probabilities, labels):
    correct_count = 0
    guess = np.argmax(probabilities, axis=1)
    for (g, l) in zip(guess, labels):
        if g == l:
            correct_count += 1
    return correct_count * 1.0 / len(labels)


def compute_correct_count(probabilities, labels):
    correct_count = 0
    guess = np.argmax(probabilities, axis=1)
    for (g, l) in zip(guess, labels):
        if g == l:
            correct_count += 1
    return correct_count,len(labels)



def run_epoch(config, model, test_data, char2id, batch_size):

    total_loss = 0
    total_correct_count = 0
    batch_num = 0

    test_set = Helper.next_seareader_model_data_batch(config, test_data, char2id, batch_size,is_for_test=True)

    for x, q, y in test_set:
        batch_num+=1
        batch_loss, attentions = model.batch_predict(x, q, y)
        correct_count, label_num = compute_correct_count(attentions, y)
        total_correct_count +=correct_count
        total_loss += batch_loss

    total_accuracy=total_correct_count *1.0/ len(test_data)
    total_loss /= batch_num
    return total_loss, total_accuracy
