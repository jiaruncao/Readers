import tensorflow as tf
import os, sys
import numpy as np
from pathlib import Path

up_two_level = str(Path(__file__).parents[2])
sys.path.append(up_two_level)
from Helper import Helper


def run(config, sess, model, test_data, char2id):
    _, test_acc,total_correct_count = run_epoch(config, model, test_data, char2id, config.test_batch_size)
    print "Test     ACC: % .5f" % (test_acc)
    print total_correct_count


def get_predict_label(probabilities):
    guess = np.argmax(probabilities, axis=1)
    return guess


def compute_accuracy(probabilities, labels):
    correct_count = 0
    guess = np.argmax(probabilities, axis=1)
    for (g, l) in zip(guess, labels):
        if g == l:
            correct_count += 1
    return correct_count, len(labels)


def run_epoch(config, model, test_data, char2id, batch_size):
    total_loss = 0
    total_correct_count = 0
    batch_num = 0

    total_guess = []
    total_label = []

    test_set = Helper.next_seareader_model_data_batch(config, test_data, char2id, batch_size, is_for_test=True)

    for x, q, y in test_set:
        batch_num += 1
        batch_loss, attentions = model.batch_predict(x, q, y)
        correct_count, label_num = compute_accuracy(attentions, y)
        total_correct_count += correct_count
        total_loss += batch_loss

        total_guess += list(get_predict_label(attentions))
        total_label+=list(y)

    display_result(test_data, total_guess, total_label)


    total_accuracy = total_correct_count * 1.0 / len(test_data)
    total_loss /= batch_num


    return total_loss, total_accuracy,total_correct_count

def display_result(test_data, total_guess, labels):
    queries = [q['query'] for q in test_data]
    result = zip(total_guess, labels)
    for q, r in zip(queries, result):
        print q, r


if __name__ == '__main__':
    pass
