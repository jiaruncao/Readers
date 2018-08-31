# -*- coding:utf-8 -*-

'''
@ author: jiarun Cao

Get the Txt File with Multi Process

'''




from multiprocessing import Process,Pool
import os,time
import xml.dom.minidom
import re
import jieba
import io
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


NUM_OF_PROCESSING = 16
MAX_FILENAME_LENGTH = 50


#进程调度，为每个进程分配文件
def process_assign(times,count):

    num_files = int (count/NUM_OF_PROCESSING)

    print 'Run child process %s (%s)' % (times, os.getpid())
    #print ('the range of file is %d -- %d' %(times*num_files) %(times+1)*num_files)
    print ("the range of file with %d process : %d -- %d"%(times,times*num_files,(times+1)*num_files))
    time.sleep(2)
    print ('----------------------------- START ----------------------------------------')
    for i in range(times*num_files,(times+1)*num_files):
        #print (i)
        if i % 10000 == 0:
            print ('--------------------- already process : %d docments ---------------------' % i)
        try:
            result = xml_preprocessing('/home/cjr/data/split_xml/' + str(i + 1) + '.xml')
            #print '%d : %s' %(i,str(result[0]))
        except Exception,e:
            print ('Error occur when preprocessing xml: %s' % str(e))

        generate_txt(result)




def xml_preprocessing(file):

    dom = xml.dom.minidom.parse(file)
    title1 = (dom.getElementsByTagName('lemmatitle'))[0].firstChild.data
    title2 = (dom.getElementsByTagName('sublemmatitle'))[0].firstChild.data
    title = title1 + '-' +title2
    if len(title) >= MAX_FILENAME_LENGTH:
        title = title[0:MAX_FILENAME_LENGTH]
    content = (dom.getElementsByTagName('content'))[0].firstChild.data
    tag1 = re.compile('</?\w+[^>]*>')
    tag2 = re.compile('\[.*?\]')
    title = tag1.sub('', title)
    title = tag2.sub('',title)
    content = tag1.sub('',content)
    content = tag2.sub('',content)
    return (title,content)

def stopwordslist(filepath):
    stopwords = [line.strip() for line in io.open(filepath, 'r',encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence = sentence.replace('\n','').replace('\t','').replace(' ','').replace('  ','')
    sentence = sentence.lower()
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords.txt')                     # load the stopwords file path here
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr



def generate_txt(result):

    try :
        fr = io.open('/home/cjr/data/txt/' + str(result[0]), 'w+', encoding='utf-8')
        fr.write(seg_sentence(result[1]))
        fr.close()
        print (str(result[0]))
    except Exception,e:
        print ('Error occur when generate txt: %s' % str(e))




if __name__ =='__main__':                                            #执行主进程

    count = 0

    for filename in os.listdir('/home/cjr/data/split_xml/'):
        count += 1
    print ('the total number of documents is %d' % count)


    '''
    # loop all files:
    for i in range(count):
        if i % 10000 == 0:
            print ('already process : %d doc' % i)
        try:
            result = xml_preprocessing('/home/cjr/data/split_xml/' + str(i + 1) + '.xml')
            generate_txt(result)
        except xml.parsers.expat.ExpatError:
            print ('xmlError occur: %d' % i)
    '''


    print ('Run the main process (%s).' % (os.getpid()))
    mainStart = time.time()                                           #记录主进程开始的时间
    p = Pool(NUM_OF_PROCESSING)                                       #开辟进程池
    for i in range(NUM_OF_PROCESSING):                                #开辟进程
        print ('GO')
        p.apply_async(process_assign,args=(i,count) )                 #每个进程都调用run_proc函数，
                                                                      #args表示给该函数传递的参数。

    print 'Waiting for all subprocesses done ...'
    p.close()                                                         #关闭进程池
    p.join()                                                          #等待开辟的所有进程执行完后，主进程才继续往下执行
    print 'All subprocesses done'
    mainEnd = time.time()                                             #记录主进程结束时间
    print 'All process ran %0.2f seconds.' % (mainEnd-mainStart)      #主进程执行时间
