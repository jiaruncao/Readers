'''
@ author: jiarun Cao

Get the Txt File
Preprocess the xml file to get the candidate documents before building indexes with lucene

'''


import xml.dom.minidom
import re
import jieba
import io
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')



#Get the title & context from a single file and remove the xml tag
def xml_preprocessing(file):

    dom = xml.dom.minidom.parse(file)
    title = (dom.getElementsByTagName('lemmatitle'))[0].firstChild.data
    content = (dom.getElementsByTagName('content'))[0].firstChild.data
    tag1 = re.compile('</?\w+[^>]*>')
    tag2 = re.compile('\[.*?\]')
    title = tag1.sub('', title)
    title = tag2.sub('',title)
    content = tag1.sub('',content)
    content = tag2.sub('',content)
    return (title,content)


# create the stopwords list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in io.open(filepath, 'r',encoding='utf-8').readlines()]
    return stopwords


# split the sentence and remove the stopwords
def seg_sentence(sentence):
    sentence = sentence.replace('\n','').replace('\t','').replace(' ','').replace('  ','')
    sentence = sentence.lower()
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords.txt')    # load the stopwords file path here
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


#generate the txt file for each xml file.
def generate_txt(result):
    try :
        fr = io.open('/home/cjr/data/txt/' + str(result[0]), 'w+', encoding='utf-8')
        fr.write(seg_sentence(result[1]))
        fr.close()
    except :
        print ('UnknowError occur: %s' %str(result[0]))


'''
# get the total number of documents
count = 0
for filename in os.listdir('/home/cjr/data/split_xml/'):
    count += 1
print ('the total number of documents is %d' %count)



# loop all files:
for i in range(73000,count):
    if i % 10000 == 0:
        print ('already process : %d doc' %i)
    try:
        result = xml_preprocessing('/home/cjr/data/split_xml/'+str(i+1)+'.xml')
        generate_txt(result)
    except :
        print ('xmlError occur: %d' %i)



#test = result[1].replace('\n','').replace('\t','').replace(' ','').replace('  ','')
#print (seg_sentence(result[1]))
#print (seg_sentence(result[0]))
'''
