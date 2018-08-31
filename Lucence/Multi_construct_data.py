#-*- coding: UTF-8 -*-
'''
@ author: jiarun Cao

Construct the training/test data for Readers

'''
import os
from multiprocessing import Pool,Process
import json
import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')




import xml.dom.minidom
import search
import Preprocess_doc


NUM_OF_PROCESSING = 16
XML_DIR = '/home/cjr/data/GKHMC.xml'    # the path of 题库.xml
TRAIN_FILENAME = '/home/cjr/data/train.json'   #the path of training data
TEST_FILENAME = '/home/cjr/data/test.json'     #the path of test data

#训练集字典：

data = {
        'query':None ,
        'answer':None  ,
        'candidates':None ,
        'documents':None
        }


def get_all_questions(file):
    print (file)
    dom = xml.dom.minidom.parse(file)
    questions = dom.getElementsByTagName('question')
    return questions



def multi_assign(times,count):
    questions = get_all_questions(XML_DIR)
    length = int(count/NUM_OF_PROCESSING)
    list_questions = []
    print ("the range of questions with %d process : %d -- %d" % (times, times * length, (times + 1) * length))

    for i in range(times*length,(times+1)*length):
        list_questions.append(questions[i])

    time.sleep(2)
    print ('----------------------------- START ----------------------------------------')

    data = multi_xml_preprocessing(list_questions)
    '''
    with open(TRAIN_FILENAME,'w+') as fr:
        for i in data:
            #print (i)

            jsob = json.dumps(i,encoding='utf-8',ensure_ascii=False)

            fr.write(jsob)
            fr.write('\n')
            #print ('okokokok')

        fr.close()
      '''
def multi_xml_preprocessing(questions):

    data_list = []
    length = len(questions)
    tag = 0
    for question in questions:
        tag += 1
        if tag % 100 == 0:
            print
        multi_question = question.getElementsByTagName('entity')


        #处理多选题
        if multi_question.length > 0 :

            query = question.getElementsByTagName('description')          #query存放题目
            data['query'] = query[0].firstChild.data
            #print(query[0].firstChild.data)
            entity = question.getElementsByTagName('entity')
            list_candidate = []
            for can in entity:                                            #list_candidate存放选项
                list_candidate.append(can.childNodes[0].nodeValue)
                #print (can.childNodes[0].nodeValue)
            #data['candidates'] = list_candidate
            candidates = question.getElementsByTagName('candidates')[0]
            candidate = candidates.getElementsByTagName('candidate')
            # 取出多选题的正确答案选项
            correct_answer = None
            for can in candidate:
                if  (can.getAttribute('value')) == str(1):
                    correct_answer = str(can.childNodes[0].nodeValue)
            data['answer'] = correct_answer[0]

            #构建candidate

            result_candidate_list = []
            for can in candidate:
                can = str(can.childNodes[0].nodeValue)
                can_option = can[0:2]
                result_candidate = str(can_option)
                can_content = [ x for x in can[2:].decode('utf-8')]
                for a in can_content:
                    for b in list_candidate:
                        if str(a.decode('utf-8')) in str(b.decode('utf-8')):
                            result_candidate += str(b[1:])
                            result_candidate += ','
                result_candidate = result_candidate.rstrip(',')

                result_candidate_list.append(result_candidate)

            #构建candidate
            can_dictize_list = []
            for can in result_candidate_list:
                can_dictize_list.append({can[0]:can[2:]})
            data['candidates'] = can_dictize_list

            '''
            correct_answer = correct_answer[2:]
            correct_answer = [x for x in str(correct_answer).decode('utf-8')]
            list_correct = []
            for i in list_candidate:
                num = str(i[0]).decode('utf-8')
                if num in correct_answer:
                    list_correct.append(i)

            for i in list_correct:
                print(i)
            '''
            # 构建文档：用题目+每个选项对应的entity作为索引，返回top 10 文档
            doc_list = []

            #for t in list_candidate:
            #    print ('the t is %s' %t[0])
            #(a,b) = [ for x,y in list_candidate]
            for can in candidate:
                search_txt = str(data['query'])
                answers = str(can.childNodes[0].nodeValue)[2:]
                answers = [t for t in str(answers).decode('utf-8')]
                for answer in answers:
                    #print (answer)
                    for a in list_candidate:
                        if answer in a.decode('utf-8'):

                            search_txt += str(a[1:].decode('utf-8'))
                        #print (x[1:])
                #print (search_txt)
                search_txt = str(Preprocess_doc.seg_sentence(search_txt))

                result = search.luceneRetriver(search_txt)
                doc_list.append(result)
            data['documents'] = doc_list



        #处理单选题
        else:
            #query存放题目
            query = question.getElementsByTagName('description')
            data['query'] = query[0].firstChild.data
            #print(query[0].firstChild.data)
            candidates = question.getElementsByTagName('candidates')[0]
            candidate = candidates.getElementsByTagName('candidate')

            #list_candidate存放选项
            list_candidate = []
            correct_answer = None
            for can in candidate:
                #print (can.firstChild.data)

                if  (can.getAttribute('value')) == str(1):
                    #correct_answer存放单选题正确答案的选项
                    correct_answer = str(can.childNodes[0].nodeValue)
                #print (can.childNodes[0].nodeValue)
                list_candidate.append(can.childNodes[0].nodeValue)


            #构建candidate
            can_dictize_list = []
            for can in list_candidate:
                can_dictize_list.append({can[0]:can[2:]})
            data['candidates'] = can_dictize_list
            #data['candidates']  = list_candidate
            data['answer'] = correct_answer[0]
            #print ('data answer: %s' %data['answer'])
            #print ('\n')
            #print (correct_answer)
            #print ('\n')

            #构建文档：用题目+每个选项作为索引，返回top 10 文档
            doc_list = []
            for can in list_candidate:
                search_txt = str(data['query']+str(can)).decode('utf-8')
                search_txt = str(Preprocess_doc.seg_sentence(search_txt))
                #lucene.initVM()
                result = search.luceneRetriver(search_txt)
                doc_list.append(result)
            data['documents'] = doc_list
        data_list.append(data.copy())
        data_add = data.copy()
        fr = open(TEST_FILENAME, 'a')
        jsob = json.dumps(data_add,encoding='utf-8',ensure_ascii=False)

        fr.write(jsob)
        fr.write('\n')
            #print ('okokokok')

        fr.close()


    return data_list



#data_ok = xml_preprocessing(XML_DIR)


'''
print ('\n')
print ('===========================')
print (str(data_ok[2]['query']))
print (len(data_ok))
for i in data_ok[2]['documents'][1]:
    print i
print (str(data_ok[2]['documents'][1]))
'''


if __name__ == '__main__':
    '''
    all_questions = get_all_questions(XML_DIR)
    print (len(all_questions))
    print ('Run the main process : %s' % os.getpid())
    mainStart =  time.time()
    p = Pool(NUM_OF_PROCESSING)
    for i in range(NUM_OF_PROCESSING):
        print ('OK')
        p.apply_async(multi_assign,args=(i,all_questions))
        print ('OVER')
    p.close()
    p.join()
    print ('All processes have been done!')
    mainEnd = time.time()
    print ('Has spent %0.2f seconds' %(mainEnd-mainStart))
    '''
    all_questions = get_all_questions(XML_DIR)
    count = len(all_questions)
    print ('Run the main process (%s).' % (os.getpid()))
    mainStart = time.time()  # 记录主进程开始的时间
    p = Pool(NUM_OF_PROCESSING)  # 开辟进程池
    for i in range(NUM_OF_PROCESSING):  # 开辟进程
        print ('GO')
        p.apply_async(multi_assign, args=(i, count))  # 每个进程都调用run_proc函数，
        # args表示给该函数传递的参数。

    print 'Waiting for all subprocesses done ...'
    p.close()  # 关闭进程池
    p.join()  # 等待开辟的所有进程执行完后，主进程才继续往下执行
    print 'All subprocesses done'
    mainEnd = time.time()  # 记录主进程结束时间
    print 'All process ran %0.2f seconds.' % (mainEnd - mainStart)  # 主进程执行时间




