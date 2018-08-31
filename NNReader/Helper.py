# -*- coding:utf-8 -*-

import csv
import pickle
import time

import gensim
import jieba
import numpy as np
import os
import pandas as pd
import json
import random, sys

from pathlib import Path

up_two_level = str(Path(__file__).parents[1])

char2id_path = '/home/cjr/data/char2id.pkl'
id2char_path = '/home/cjr/data/id2char.pkl'


# old version, too slow for big file
def get_embedding_txt(embedding_path=None):
    """
    load embedding, <UNk> and <PAD> are 0 vector
    :param infile_path: emb file path, one row represent a emb, word x x x ...: separated by space
    :return: emb matrix: ndarray
    """
    global emb_matrix
    if os.path.isfile(char2id_path) and os.path.isfile(id2char_path):
        char2id = pickle.load(open(char2id_path, "rb"))
        id2char = pickle.load(open(id2char_path, "rb"))
    else:
        char2id, id2char = build_map(embedding_path)

    row_index = 0

    if not embedding_path:
        print 'no embedding!'
        return 0

    with open(embedding_path, "rb") as infile:
        for row in infile:
            row = row.strip()
            row_index += 1
            # first line denotes the number of words and the emb dimension
            if row_index == 1:
                num_chars = int(row.split()[0])
                emb_dim = int(row.split()[1])
                emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
                continue
            items = row.split()

            char = items[0]

            emb_vec = [float(val) for val in items[1:]]

            if char.decode('utf-8') in char2id.keys():
                emb_matrix[char2id[char.decode('utf-8')]] = emb_vec
    return emb_matrix, char2id, id2char


def build_map_txt(embedding_path):
    """
    Construct char2id and id2char
    """
    # remove the first line
    df_emb = pd.read_csv(embedding_path, sep=' ', encoding='utf-8', usecols=[0], names=['chars'], skiprows=[0],
                         quoting=csv.QUOTE_NONE)
    chars = list(set(df_emb.chars))
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))

    id2char[0] = "<PAD>"
    char2id["<PAD>"] = 0
    id2char[len(chars) + 1] = "<UNK>"
    char2id["<UNK>"] = len(chars) + 1

    save_map(id2char, char2id)

    return char2id, id2char


def save_map(id2char, char2id):
    pickle.dump(char2id, open(char2id_path, 'w'))
    pickle.dump(id2char, open(id2char_path, 'w'))
    print "saved map between char and id"


def generate_alter_model_reader_input(data, char2id):
    """
    Generate data for alter model
    Padding and covert words to index
    """
    doc_max_len = 0
    query_max_len = 0
 
    X = []
    Xq = []
    Y = []
    for l in data:
        s = l['document']
        q = l['query']
        a = l['answer']

        # concat all documents into one document
        s = reduce(lambda x, y: x + y, s)

        # get max length
        doc_max_len = max(len(s), doc_max_len)
        query_max_len = max(len(q), query_max_len)

        x = words2id_docs(s, char2id)
        xq = words2id_statement(q, char2id)

        X.append(x)
        Xq.append(xq)
        Y.append(char2id[a])

    X = pad_sequences(X, maxlen=doc_max_len)
    Q = pad_sequences(Xq, maxlen=query_max_len)
    return (X, Q, np.array(Y))


def read_json_file(json_file_path):
    # load from raw data of our project
    # load pkl file
    if os.path.splitext(json_file_path)[1] == '.pkl':
        return pickle.load(open(json_file_path, "rb"))
    # load json file
    elif os.path.splitext(json_file_path)[1] == '.json':
        data_list = []
        with open(json_file_path) as f:
            for line in f:
                data_list.append(json.loads(line))
        return data_list


# padding, from keras
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def words2id_docs(X, char2id):
    X_id = []
    for instance in X:
        X_id.append(instance2id(instance, char2id))

    return X_id


def words2id_statement(X, char2id):
    X_id = instance2id(X, char2id)

    return X_id


def words2id_recursive(X, char2id):
    """
    convert char list to id list, can handle nested lists
    :param X: char list
    :param char2id: dict
    :return: a list
    """
    X_id = list()
    for x in X:
        if type(x) == list:
            X_id.append(words2id_recursive(x, char2id))
        else:
            if type(x) == str:
                x = x.decode('utf-8')
            if x in char2id.keys():
                X_id.append(char2id[x])
            else:
                X_id.append(char2id[u'<UNK>'])
    return X_id


def filename2content(filename_list, file_path, doc_max_len):
    """convert document name to document content"""
    # each list corresponds for a document
    document = []
    for f in filename_list:
        # file name, score
        f, _ = f
        f_path = os.path.join(file_path, f)
        content = []
        if os.path.exists(f_path):
            with open(f_path, 'r') as f_r:
                for l in f_r:
                    # The separator between word and word is a space
                    l = l.split(' ')
                    if len(l) >= doc_max_len:
                        content += l[:doc_max_len]
                        break
                    else:
                        content += l
                        doc_max_len = doc_max_len - len(l)
        # else:
        #     print "Document not found：", f
        document.append(content)
    return document


def keep_top_N_from_all_document(s, n):
    """merge all retrieved document for all answer, and keep top N document"""
    # merge all documents
    s_list = []
    for d in s:
        s_list += d
    merged_dict = merge_two_dicts(s_list[0], s_list[1])
    for d in s_list[2:]:
        merged_dict = merge_two_dicts(merged_dict, d)

    # return n file names. A list
    return sorted(merged_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[:n]


def keep_top_N_document(s, n):
    """keep top N document retrieved from one answer"""
    s_top_n = []
    for answer_docs in s:
        merged_dict = merge_two_dicts(answer_docs[0], answer_docs[1])
        for d in answer_docs[2:]:
            merged_dict = merge_two_dicts(merged_dict, d)
        s_top_n.append(sorted(merged_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[:n])

    return s_top_n


def generate_statement(query, candidates):
    """concat each candidate with query"""
    statement = []
    for c in candidates:
        statement.append(query + c.values()[0])
    return statement


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def cut_word(s):
    s_cut = []
    for i in s:
        if type(i) == list:
            s_cut.append(cut_word(i))
        else:
            s_cut.append([j for j in jieba.cut(i)])
    return s_cut


def answer2id(answer):
    """convert answer to index so that it can be turned into one-hot"""
    d_a = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    if answer in d_a.keys():
        return d_a[answer]
    else:
        print 'wrong answer!'


# def next_seareader_sharedoc_model_data_batch(config, data_list, char2id, batch_size):
#     """
#     Generate data for seareader model.
#     Padding and covert words to index
#     :param data_list: data is a nested list, read from json
#     :param char2id: dict
#     :return:
#     """
#     # Shuffle data
#     random.shuffle(data_list)
#
#     # whether finish a turn
#     flag = False
#
#     X = []
#     Xq = []
#     Y = []
#
#     i = 0
#     while True:
#         if i == len(data_list):
#             i = 0
#             flag = True
#
#         # list-->4 elements-->10 elements-->docu name
#         document = data_list[i]['documents']
#         # string
#         q = data_list[i]['query']
#         # string, ABCD
#         a = data_list[i]['answer']
#         # list-->dict
#         c = data_list[i]['candidates']
#
#         # keep the top-N documents by document score
#         document = keep_top_N_from_all_document(document, config.top_n)
#         # convert document name to content
#         document = filename2content(document, config.document_save_path, config.doc_max_len)
#
#         # concat query and answer
#         statement = generate_statement(q, c)
#         # Word Segmentation
#         statement = cut_word(statement)
#
#         x = words2id_docs(document, char2id)
#         xq = words2id_statement(statement, char2id)
#
#         X.append(x)
#         Xq.append(xq)
#         Y.append(answer2id(a))
#
#         if len(X) == batch_size:
#             pad_X = []
#             pad_Q = []
#             for x in X:
#                 pad_X.append(pad_sequences(x, maxlen=config.doc_max_len))
#             for xq in Xq:
#                 pad_Q.append(pad_sequences(xq, maxlen=config.statement_max_len))
#             # document, statement, label
#             yield (pad_X, pad_Q, np.array(Y))
#
#             # restart
#             X = []
#             Xq = []
#             Y = []
#
#             if flag:
#                 break
#
#         i += 1


def next_seareader_model_data_batch(config, data_list, char2id, batch_size, is_for_test=False):
    """
    A generator. Generate batches neural_reasoner_model.
    :param statement_max_len: max statement length
    :param doc_max_len: max doc length
    :param top_n: top n documents for each answer
    :param document_save_path: documents storage path
    :param data_list: data is a nested list, read from json
    :param char2id: dict
    :return: Documents, query+answer, labels
    """
    if not is_for_test:
        # Shuffle data
        random.shuffle(data_list)

    # whether finish a turn
    flag = False

    X = []
    Xq = []
    Y = []

    i = 0

    while True:


        # list-->4 elements-->10 elements-->docs name
        document = data_list[i]['documents']
        # string
        q = data_list[i]['query']
        # string, ABCD
        a = data_list[i]['answer']
        # list-->dict
        c = data_list[i]['candidates']

        # keep the top-N documents by document score for each answer
        document = keep_top_N_document(document, config.top_n)

        # convert document name to content
        docs_content = []
        for d in document:
            docs_content.append(filename2content(d, config.document_save_path, config.doc_max_len))

        # concat query and answer
        statement = generate_statement(q, c)
        # Word Segmentation
        statement = cut_word(statement)

        x = words2id_docs(docs_content, char2id)
        xq = words2id_statement(statement, char2id)

        X.append(x)
        Xq.append(xq)
        Y.append(answer2id(a))

        if i == len(data_list)-1:
            i = 0
            flag = True
        else:
            i += 1

        # for train data and test data
        if len(X) == batch_size or (flag and is_for_test):
            pad_X = []
            pad_Q = []

            for x in X:
                pad_xi = []
                for xi in x:
                    pad_xi.append(pad_sequences(xi, maxlen=config.doc_max_len))
                pad_X.append(pad_xi)
            for xq in Xq:
                pad_Q.append(pad_sequences(xq, maxlen=config.statement_max_len))

            # document, statement, label
            yield (pad_X, pad_Q, np.array(Y))

            # restart
            X = []
            Xq = []
            Y = []

            if flag:
                break




def next_neural_reasoner_model_data_batch(data_list, char2id, batch_size, top_n, doc_max_len, statement_max_len,
                                          document_save_path,is_for_test=False):
    """
    A generator. Generate batches neural_reasoner_model.
    :param statement_max_len: max statement length
    :param doc_max_len: max doc length
    :param top_n: top n documents for each answer
    :param document_save_path: documents storage path
    :param data_list: data is a nested list, read from json
    :param char2id: dict
    :return: Documents, query+answer, labels
    """
    if not is_for_test:
        # Shuffle data
        random.shuffle(data_list)

    # whether finish a turn
    flag = False

    X = []
    Xq = []
    Y = []

    i = 0

    while True:


        # list-->4 elements-->10 elements-->docs name
        document = data_list[i]['documents']
        # string
        q = data_list[i]['query']
        # string, ABCD
        a = data_list[i]['answer']
        # list-->dict
        c = data_list[i]['candidates']

        # keep the top-N documents by document score for each answer
        document = keep_top_N_document(document, top_n)

        # convert document name to content
        docs_content = []
        for d in document:
            docs_content.append(filename2content(d, document_save_path, doc_max_len))

        # concat query and answer
        statement = generate_statement(q, c)
        # Word Segmentation
        statement = cut_word(statement)

        x = words2id_docs(docs_content, char2id)
        xq = words2id_statement(statement, char2id)

        X.append(x)
        Xq.append(xq)
        Y.append(answer2id(a))

        if i == len(data_list)-1:
            i = 0
            flag = True
        else:
            i += 1

        # for train data and test data
        if len(X) == batch_size or (flag and is_for_test):
            pad_X = []
            pad_Q = []

            for x in X:
                pad_xi = []
                for xi in x:
                    pad_xi.append(pad_sequences(xi, maxlen=doc_max_len))
                pad_X.append(pad_xi)
            for xq in Xq:
                pad_Q.append(pad_sequences(xq, maxlen=statement_max_len))

            # document, statement, label
            yield (pad_X, pad_Q, np.array(Y))

            # restart
            X = []
            Xq = []
            Y = []

            if flag:
                break

def get_embedding(embedding_path):
    """
    load embedding, <UNk> and <PAD> are 0 vector
    :param infile_path: emb file path, one row represent a emb, word x x x ...: separated by space
    :return: emb matrix: ndarray
    """
    model = gensim.models.Word2Vec.load(embedding_path)

    if os.path.isfile(char2id_path) and os.path.isfile(id2char_path):
        char2id = pickle.load(open(char2id_path, "rb"))
        id2char = pickle.load(open(id2char_path, "rb"))
    else:
        char2id, id2char = build_map(model)

    emb_matrix = model.wv.syn0
    zero_vec = np.zeros(emb_matrix.shape[1])
    # add <PAD>
    emb_matrix = np.insert(emb_matrix, 0, values=zero_vec, axis=0)
    # add <UNK>
    emb_matrix = np.insert(emb_matrix, len(emb_matrix), values=zero_vec, axis=0)

    return emb_matrix, char2id, id2char


def build_map(model):
    """
    Construct char2id and id2char
    """
    chars = model.wv.index2word

    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))

    id2char[0] = u"<PAD>"
    char2id[u"<PAD>"] = 0
    id2char[len(chars) + 1] = u"<UNK>"
    char2id[u"<UNK>"] = len(chars) + 1

    save_map(id2char, char2id)

    return char2id, id2char


def instance2id(instance, wor2id):
    return [sentence2id(sent, wor2id) for sent in instance]


def sentence2id(sentence, word2id):
    return [turn_word2id(w, word2id) for w in sentence]


def turn_word2id(word, word2id):
    if type(word) == str:
        word = word.decode('utf-8')
    return word2id.get(word, word2id[u'<UNK>'])


if __name__ == '__main__':
    next_seareader_model_data_batch()
