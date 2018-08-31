import os, sys, glob



DIRTOINDEX = '/home/elephant/Documents/CJR-Gaokao/Gaokao/txt/'

'''
for tfile in glob.glob(os.path.join(DIRTOINDEX, '*.txt')):
    print "Indexing: ", tfile
    print ('okokokook')
'''


list = os.listdir(DIRTOINDEX)
for i in range(len(list)):
    path = os.path.join(DIRTOINDEX,list[i])
    if os.path.isfile(path):
        print (path)