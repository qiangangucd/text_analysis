import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


# Start Q1
def Q1():
    data = open('text').read()
    # nltk.download('punkt')
    data_token = nltk.word_tokenize(data)
    data_normalisation = [i.lower() for i in data_token]
    print(data_normalisation)
    # nltk.download('averaged_perceptron_tagger')
    postagged = nltk.pos_tag(data_normalisation)
    print(postagged)


def Q2():
    data = open('text').read()
    data_token = nltk.word_tokenize(data)
    stemmer = PorterStemmer()
    stemed = [stemmer.stem(val) for val in data_token]
    print(stemed)
    stemmer = nltk.LancasterStemmer()
    stemed = [stemmer.stem(val) for val in data_token]
    print(stemed)
    stemmer = SnowballStemmer('english')
    stemed = [stemmer.stem(val) for val in data_token]
    print(stemed)
    postagged = nltk.pos_tag(data_token)
    wn = nltk.WordNetLemmatizer()
    wn.lemmatize()


if __name__ == '__main__':
    Q1()
    Q2()
