#-*- coding: UTF-8 -*-
#!/usr/bin/env python


"""
@ author: jiarun Cao

PyLucene retriver simple example
"""

import lucene
from lucene import SimpleFSDirectory, System, File, Document, Field, \
    StandardAnalyzer, IndexSearcher, Version, QueryParser

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


INDEXDIR = '/home/cjr/data/index/'  #the index file path
TXTDIR = '/home/cjr/data/txt/'      #the documents path (txt file)


#search for a query with pylucene
def luceneRetriver(query):
    #print ('-------------Searching-------------')
    #print (query)
    lucene.initVM()
    indir = SimpleFSDirectory(File(INDEXDIR))
    lucene_analyzer = StandardAnalyzer(Version.LUCENE_30)
    lucene_searcher = IndexSearcher(indir)
    my_query = QueryParser(Version.LUCENE_30, 'text',lucene_analyzer).parse(query)
    MAX = 1000

    #存放返回的文档标题list
    title_list = []

    total_hits = lucene_searcher.search(my_query, MAX)

    #print "Hits: ", total_hits.totalHits

    for hit in total_hits.scoreDocs[:10]:

        #print"Hit Score: ", hit.score, "Hit Doc:", hit.doc, "HitString:", hit.toString()

        doc = lucene_searcher.doc(hit.doc)

        #print doc.get("title").encode("utf-8").lstrip(str(TXTDIR))
        #print doc.get("text").encode("utf-8")
        #print ('\n')

        title_list.append({doc.get("title").encode("utf-8").lstrip(str(TXTDIR)):round(hit.score,5)}.copy())

    return title_list
#print ('查询内容：八卦')
#print ('查询结果:')
#print ('\n')
#luceneRetriver("下列 关于 中国 八卦 不正确 人类 历史 东西方 平等 交流 见证")