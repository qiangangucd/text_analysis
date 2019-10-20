import math
from collections import Counter

vector_dict = {}


# Just loads in all the documents
def load_docs():
    print("Loading docs...")
    # doc1 = ('d1', 'LSI tutorials and fast tracks')
    # doc2 = ('d2', 'books on semantic analysis')
    # doc3 = ('d3', 'learning latent semantic indexing')
    # doc4 = ('d4', 'advances in structures and advances in indexing')
    # doc5 = ('d5', 'analysis of latent structures')
    # return [doc1, doc2, doc3, doc4, doc5]
    # doc1=('d1', open('data/doc1','r').read())
    # doc2 = ('d2', open('data/doc2', 'r').read())
    # doc3 = ('d3', open('data/doc3', 'r').read())
    # doc4=('d4',open('data/doc1ed','r').read())
    doc1=('d1','Find 3 short documents about which you might want to know their similarity.')
    doc2=('d2','Produce 5 variants on one of the documents and see how the cosine similarity changes.')
    doc3=('d3','Find a python package that computes cosine similarity and euclidean distance.')
    doc1_01=('d1_01','Find 3 short documents about which you may want to know their similarity.')
    doc1_02=('d1_02','Found 3 short documents about which you might want to know their similarity.')
    doc1_03=('d1_03','Find 3 short document about which you might want to know their similarity.')
    doc1_04=('d1_04','Find 3 short documents about which you might want to know their similarity the.')
    doc1_05=('d1_05','Find 3 short documents about which you might want to know their similarities.')
    return [doc1,doc2,doc3,doc1_01,doc1_02,doc1_03,doc1_04,doc1_05]



# Computes TF for words in each doc, DF for all features in all docs; finally whole Tf-IDF matrix
def process_docs(all_dcs):
    stop_words = ['of', 'and', 'on', 'in','to','you','a','the']
    all_words = []
    counts_dict = {}
    for doc in all_dcs:
        words = [x.lower() for x in doc[1].split() if x not in stop_words]
        words_counted = Counter(words)
        unique_words = list(words_counted.keys())
        counts_dict[doc[0]] = words_counted
        all_words = all_words + unique_words
    n = len(counts_dict)
    df_counts = Counter(all_words)
    compute_vector_len(counts_dict, n, df_counts)


# computes TF-IDF for all words in all docs
def compute_vector_len(doc_dict, no, df_counts):
    global vector_dict
    for doc_name in doc_dict:
        doc_words = doc_dict[doc_name].keys()
        wd_tfidf_scores = {}
        for wd in list(set(doc_words)):
            wds_cts = doc_dict[doc_name]
            wd_tf_idf = wds_cts[wd] * math.log(no / df_counts[wd], 10)
            wd_tfidf_scores[wd] = round(wd_tf_idf, 4)
        vector_dict[doc_name] = wd_tfidf_scores


def get_cosine(text1, text2):
    vec1 = vector_dict[text1]
    vec2 = vector_dict[text2]
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return round(float(numerator) / denominator, 3)


# RUN
all_docs = load_docs()
process_docs(all_docs)
# vector_dict['q'] = {'semantic': 1, 'latent': 1, 'indexing': 1}

for keys, values in vector_dict.items(): print(keys, values)
#
# text1 = 'd3'
# print(text1)
# text2 = 'q'
cosine=[]
text1='d1'
text2='d2'
text3='d3'
text1_01='d1_01'
text1_02='d1_02'
text1_03='d1_03'
text1_04='d1_04'
text1_05='d1_05'
cosine.append(get_cosine(text1, text2))
cosine.append(get_cosine(text1, text3))
cosine.append(get_cosine(text1, text1_01))
cosine.append(get_cosine(text1, text1_02))
cosine.append(get_cosine(text1, text1_03))
cosine.append(get_cosine(text1, text1_04))
cosine.append(get_cosine(text1, text1_05))
cosine.append(get_cosine(text2, text3))
cosine.append(get_cosine(text2, text1_01))
cosine.append(get_cosine(text2, text1_02))
cosine.append(get_cosine(text2, text1_03))
cosine.append(get_cosine(text2, text1_04))
cosine.append(get_cosine(text2, text1_05))
cosine.append(get_cosine(text3, text1_01))
cosine.append(get_cosine(text3, text1_02))
cosine.append(get_cosine(text3, text1_03))
cosine.append(get_cosine(text3, text1_04))
cosine.append(get_cosine(text3, text1_05))
cosine.append(get_cosine(text1_01, text1_02))
cosine.append(get_cosine(text1_01, text1_03))
cosine.append(get_cosine(text1_01, text1_04))
cosine.append(get_cosine(text1_01, text1_05))
cosine.append(get_cosine(text1_02, text1_03))
cosine.append(get_cosine(text1_02, text1_04))
cosine.append(get_cosine(text1_02, text1_05))
cosine.append(get_cosine(text1_03, text1_04))
cosine.append(get_cosine(text1_03, text1_05))
cosine.append(get_cosine(text1_04, text1_05))
print(len(cosine))
print('Cosine:', cosine)


import matplotlib.pyplot as plt
import numpy as np

# plt.scatter(range(3),[0.016, 0.018, 0.04],color='r',label='before produced variants')
plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(range(28),cosine)
plt.xlabel('Pair')
plt.ylabel('Cosine similarity')
labels=['(d1,d2)','(d1,d3)','(d1,d1_01)','(d1,d1_02)','(d1,d1_03)','(d1,d1_04)','(d1,d1_05)','(d2,d3)','(d2,d1_01)','(d2,d1_02)','(d2,d1_03)','(d2,d1_04)','(d2,d1_05)','(d3,d1_01)','(d3,d1_02)','(d3,d1_03)','(d3,d1_04)','(d3,d1_05)','(d1_01,d1_02)','(d1_01,d1_03)','(d1_01,d1_04)','(d1_01,d1_05)','(d1_02,d1_03)','(d1_02,d1_04)','(d1_02,d1_05)','(d1_03,d1_04)','(d1_03,d1_05)','(d1_03,d1_05)']
plt.xticks(range(28),labels,rotation='vertical')
plt.savefig('diff.png',dpi=100,bbox_inches='tight')
plt.show()
