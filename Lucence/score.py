#-*- coding: UTF-8 -*-
import json
import os
TOP_N = 3
def read_json_file(json_file_path):
    # load from raw data of our project
    data_list = []
    with open(json_file_path) as f:
        for line in f:
            data_list.append(json.loads(line))
    return data_list

if __name__ == '__main__':
    data_list = read_json_file('/home/elephant/Documents/CJR-Gaokao/Gaokao/test.json')
    sum_score_list = []
    answer_list= []
    for data in data_list:
        answer_list.append(data['answer'])
        score_list = []
        for i in range(len(data['documents'])):

            score = 0.0
            for j in range(TOP_N):
                score += float(data['documents'][i][j].values()[0])
            score_list.append(score)
        sum_score_list.append(score_list)
    flag_list = []
    for score in sum_score_list:
        max_index = score.index(max(score))
        FLAG = None
        if max_index == 0 :
            FLAG = 'A'
        if max_index == 1 :
            FLAG = 'B'
        if max_index == 2 :
            FLAG = 'C'
        if max_index == 3 :
            FLAG = 'D'
        flag_list.append(FLAG)
    correct = 0
    #accuracy = 0.0
    for i in range(len(flag_list)):
        a = flag_list[i]
        b = answer_list[i].encode('unicode-escape').decode('string-escape')
        if a == b :
            correct += 1
    print (correct)
    print (len(flag_list))
    accuracy = float(float(correct)/float(len(flag_list)))
    print ('Test acc is %.2f' %accuracy)


