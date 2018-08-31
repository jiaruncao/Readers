'''
@ author: jiarun Cao

Split the big xml file to small fragment

'''



import re
with open('/home/cjr/data/BaikeAll.xml') as fr:
    dummy = 0
    count = 1
    for line in fr:
        if dummy < 2:
            dummy += 1
            continue

        if '<item>' in line and '</item>' not in line :
            fr_2 = open('/home/cjr/data/split_xml/'+str(count)+'.xml','w+')
            fr_2.write(line)
        elif '<item>' not in line and '</item>' not in line:
            fr_2.write(line)
        elif '<item>'  in line and '</item>'  in line:
            fr_2.write(line)
            fr_2.close()
            count += 1
        elif '<item>' not in line and '</item>' in line:
            fr_2.write(line)
            fr_2.close()
            count += 1


        if count % 1000 == 0 :
            print (count)
