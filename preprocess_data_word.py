
import pandas as pd
import numpy as np
from datetime import datetime



'''
read_data
'''
path = '/Users/Documents/dataset/'
airline = pd.read_csv(path + 'airline.csv')
cache_map = pd.read_csv(path + 'cache_map.csv')
day_schedule = pd.read_csv(path + 'day_schedule.csv')
group = pd.read_csv(path + 'group.csv')
order = pd.read_csv(path + 'order.csv')
keyword = pd.read_csv(path + 'key_word-1.csv' , header = None)


'''
airline_modify
'''
airline['go_back'] = pd.get_dummies(airline['go_back'])['去程']
airline['src_airport'] = airline['src_airport'].str.split().str[0]
airline['dst_airport'] = airline['dst_airport'].str.split().str[0]
airline['fly_time'] = airline['fly_time'].map(lambda a: datetime.strptime(a,'%Y/%m/%d %H:%M'))
airline['arrive_time'] = airline['arrive_time'].map(lambda a: datetime.strptime(a,'%Y/%m/%d %H:%M'))
airline['fly_month'] = airline['fly_time'].map(lambda a: a.month)
airline['arrive_month'] = airline['arrive_time'].map(lambda a: a.month)
airline['fly_weekday'] = airline['fly_time'].map(lambda a: a.weekday())
airline['arrive_weekday'] = airline['arrive_time'].map(lambda a: a.weekday())
airline['travel_time'] = airline['fly_time'] - airline['arrive_time']
airline['travel_time'] = airline['travel_time'].map(lambda a: a.seconds/3600)

'''
group_promotion
'''
group['promotion_all'] = group['product_name'] + group['promotion_prog']

import jieba.analyse
jieba.set_dictionary("jieba/dict.txt")
seg_list = jieba.cut("在非洲，每六十秒，就有一分鐘過去") 
print("|".join(seg_list))

import re
import string
exclude = set(string.punctuation)

def getChinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = re.sub('微軟正黑體','', context)
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    return context

def get_discount(context):
    position = 0
    if context.find('省') != -1:
        position = context.find('省')
    elif context.find('扣') != -1:
        position = context.find('扣')
    elif context.find('折') != -1:
        position = context.find('折')
    elif context.find('惠') != -1:
        position = context.find('惠')
    elif context.find('減') != -1:
        position = context.find('減')
    else:
        position = 'nan'
        
    discount = ''
    while True:
        if position == 'nan':
            discount = '0'
            break
        else:
            position+=1
            if context[position].isdigit():
                discount+=context[position]
            elif context[position] == '千':
                discount+='000'
            elif context[position] == '$':
                pass
            elif context[position]=='＄':
                pass
            else:
                break
    if len(discount) == 0:
        discount = '0'
    return discount

def remove_punc(context):
    context = re.sub("[\s+\.\!\/_,$%^*( [ ]  ‧ +\"\']+|[+——！，。？、)~@#￥%……&*（）．＋+]+", "",context)
    context = re.sub("[【】╮╯▽╰╭★→ ‧「」]+","",context)
    context = re.sub("[！，❤。～《》：（）◆ ＄ ●【】‧ []「」？”“；：、]","",context)
    context_ = ''.join(ch for ch in context if ch not in exclude)
    return context_
        
        
for index, row in enumerate(group['product_name']):
    if pd.isnull(row):
        group['product_name'][index] = ''
    else:
        print(index)
        group['product_name'][index] = getChinese(row)
        
for index, row in enumerate(group['promotion_prog']):
    if pd.isnull(row):
        group['product_name'][index] = ''
    else:
        print(index)
        group['product_name'][index] = remove_punc(row)
        
discount_list = []
for index, row in enumerate(group['product_name']):
    if pd.isnull(row):
        discount_list.append('0')
    else:
        print(index)
        discount = get_discount(row)
        discount_list.append(int(discount))
        
document=[]            
for index, row in enumerate(group['product_name']):
    seg_list = jieba.cut(row)
    result = []
    for seg in seg_list :
        seg = ''.join(seg.split())
        if (seg != '' and seg != "\n" and seg != "\n\n" and seg != "日" and seg != "五日" and seg != "四日") :
            result.append(seg)
    document.append(result)
    

from gensim import corpora

dictionary = corpora.Dictionary(document)
corpus = [dictionary.doc2bow(text) for text in document]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


import gensim
NUM_TOPICS = 50
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(50,num_words=10)
for topic in topics:
    print(topic)

answer = []
for doc in corpus:
    answer.append(ldamodel.get_document_topics(doc))
    
matrix = []
for line in answer:
    weight_list = []
    for i in range(NUM_TOPICS):
        tup = [item for item in line if i in item]
        if len(tup) == 0:
            weight_list.append(0)
        else:
            weight_list.append(float(tup[0][1]))
    matrix.append(weight_list)
        
matrix = np.asarray(matrix)     
discount_list = np.asarray(discount_list, dtype=np.float32)
y = discount_list.reshape((-1,1))
z = np.append(matrix,y,axis=1)

import numpy
numpy.savetxt("topics_model.csv", matrix, delimiter=",")
numpy.savetxt("discount.csv", y, delimiter=",")

key_list = []        
for row in keyword[0]:
    print(row)
    key = []
    key = list(group['product_name'].str.contains(row))
    key[6443] = False
    key[6520] = False
    key[6570] = False
    key[30627] = False
    key[37874] = False
    key = [int(x) for x in key]
    key_list.append(key)
        
key_array = np.asarray(key_list).T
numpy.savetxt("key_word_array.csv", key_array, delimiter=",")
