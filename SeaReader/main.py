import pprint
import tensorflow as tf
import os
from datetime import datetime

import sys
from pathlib import Path

up_two_level = str(Path(__file__).parents[2])

sys.path.append(up_two_level)
from Helper import Helper

from SeaReader import SeaReader
import train
import test

flags = tf.app.flags
flags.DEFINE_string("gpu", "1", "Gpu number (default: 3)")

flags.DEFINE_integer("answer_num", 4, "Number of answer (default: 4)")
flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 384)")
flags.DEFINE_integer("hidden_dim", 128, "Dimensionality of hidden layer")
flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0001)")
flags.DEFINE_float("learning_rate", 1e-3, "AdamOptimizer learning rate (default: 0.001)")
flags.DEFINE_float("learning_rate_decay", 0.8,
                   "How much learning rate will decay after half epoch of non-decreasing loss (default: 0.8)")

flags.DEFINE_integer("doc_max_len", 100, "Max length of a document (default: 200)")
flags.DEFINE_integer("statement_max_len", 60, "Max length of a statement (default: 100)")
flags.DEFINE_integer("top_n", 5, "Number of documents (default: 5)")

# Training parameters
flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 32)")
flags.DEFINE_integer("evaluate_num", 1000, "Number of evaluate set (default: 200)")

flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 12)")
flags.DEFINE_integer("evaluate_every", 40, "Evaluate model on validation set after this many steps (default: 300)")

# flags.DEFINE_boolean("trace", False, "Trace (load smaller dataset)")
flags.DEFINE_string("log_dir", "logs", "Directory for summary logs to be written to default (./logs/)")

flags.DEFINE_integer("checkpoint_every", 50, "Run on valid set")
flags.DEFINE_string("ckpt_dir", up_two_level + "/data/SeaReader_ckpts_1000_zuizhong3/",
                    "Directory for checkpoints default (../../data/SeaReader_ckpts/)")
flags.DEFINE_string("restore_file", "/home/xiaoxinglin/GaoKaoReader/data/SeaReader_ckpts_test/model-l2.611_a0.548.ckpt-3562", "Checkpoint to load")
# flags.DEFINE_string("document_save_path", "/home/xiao/Desktop/GaoKaoReaderData/txt/", "document storage path")
flags.DEFINE_string("document_save_path", "/home/xiaoxinglin/txt/", "document storage path")
flags.DEFINE_boolean("evaluate", False, "Whether to run evaluation epoch on a checkpoint. Must have restore_file set.")

# debug
flags.DEFINE_boolean("debug_run", False, "Debug mode, load a small train set")
flags.DEFINE_integer("debug_batch_size", 100, "Batch Size")
flags.DEFINE_integer("debug_train_num", 10000, "Debug train size")
flags.DEFINE_integer("debug_evaluate_batch_size", 5,
                     "Evaluate Batch Size, must be divisible by the length of the validation set (default: 32)")
flags.DEFINE_integer("debug_evaluate_num", 5, "Evaluate num")
flags.DEFINE_integer("debug_checkpoint_every", 2, "Run on valid set")

# file path
flags.DEFINE_string("embedding_path", "/home/xiaoxinglin/GaoKaoReader/data/Readers_word2vec", "Path of embedding")
flags.DEFINE_string("train_json_path", "/home/xiaoxinglin/GaoKaoReader/data/train.pkl", "Path of train json data")
flags.DEFINE_string("test_json_path", "/home/xiaoxinglin/GaoKaoReader/data/test.json", "Path of test json data")


def main(_):
    FLAGS = tf.app.flags.FLAGS
    pp = pprint.PrettyPrinter()
    FLAGS._parse_flags()
    pp.pprint(FLAGS.__flags)

    # Load embedding
    emb_matrix, char2id, id2char = Helper.get_embedding(FLAGS.embedding_path)
    print "Load embedding"

    # Directly load data into list
    train_data_list = Helper.read_json_file(FLAGS.train_json_path)
    print 'Train data num:',len(train_data_list)
    test_data_list = Helper.read_json_file(FLAGS.test_json_path)
    # test_data_list =None

    # Create model storage directories
    if not os.path.exists(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)

    timestamp = datetime.now().strftime('%c')
    FLAGS.log_dir = os.path.join(FLAGS.log_dir, timestamp)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # Gpu number
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Train Model
    with tf.Session(config=config) as sess:
        model = SeaReader(FLAGS.doc_max_len, FLAGS.top_n, FLAGS.statement_max_len, FLAGS.hidden_dim,
                          FLAGS.answer_num, FLAGS.embedding_dim, emb_matrix, FLAGS.learning_rate, sess)
        saver = tf.train.Saver(max_to_keep=50)

        # Run evaluation
        if FLAGS.evaluate:
            print '[?] Test run'
            if not FLAGS.restore_file:
                print('Need to specify a restore_file checkpoint to evaluate')
            else:
                print('[?] Loading variables from checkpoint %s' % FLAGS.restore_file)
                saver.restore(sess, FLAGS.restore_file)
                test.run(FLAGS, sess, model, test_data_list, char2id)
        elif FLAGS.debug_run:
            print '[?] Debug run'
            train.debug_run(FLAGS, sess, model, train_data_list, test_data_list, char2id, saver)
        else:
            print '[?] Run'
            train.run(FLAGS, sess, model, train_data_list, test_data_list, char2id, saver)


if __name__ == '__main__':
    tf.app.run()
