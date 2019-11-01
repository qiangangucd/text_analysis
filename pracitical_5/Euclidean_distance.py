from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

doc1 = 'Find 3 short documents about which you might want to know their similarity.'
doc2 = 'Produce 5 variants on one of the documents and see how the cosine similarity changes.'
doc3 = 'Find a python package that computes cosine similarity and euclidean distance.'
doc1_01 = 'Find 3 short documents about which you may want to know their similarity.'
doc1_02 = 'Found 3 short documents about which you might want to know their similarity.'
doc1_03 = 'Find 3 short document about which you might want to know their similarity.'
doc1_04 = 'Find 3 short documents about which you might want to know their similarity the.'
doc1_05 = 'Find 3 short documents about which you might want to know their similarities.'
documents = [doc1, doc2, doc3, doc1_01, doc1_02, doc1_03, doc1_04, doc1_05]

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
                  columns=count_vectorizer.get_feature_names(),
                  index=[doc1, doc2, doc3, doc1_01, doc1_02, doc1_03, doc1_04, doc1_05])
print(euclidean_distances(df, df))
