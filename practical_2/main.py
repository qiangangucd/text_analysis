import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import urllib.request
import bs4
import re


# Start Q1
def q_1(data):
    data = text_clean(data)
    data_token = nltk.word_tokenize(data)
    print('data_token=', data_token)
    data_normalisation = [word.lower() for word in data_token]
    print('data_normalisation=', data_normalisation)
    # nltk.download('averaged_perceptron_tagger')
    postagged = nltk.pos_tag(data_normalisation)
    print('postagged=', postagged)
    # print(' '.join(data_token))


def text_clean(data):
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', data)
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    text_no_small_words = re.sub(r'\b\w{1,2}\b', '', text_no_hyperlinks)
    text_no_whitespace = re.sub(r'\s\s+', ' ', text_no_small_words)
    text_no_whitespace = text_no_whitespace.lstrip(' ')
    text_no_punctuation = re.sub(r'\W', ' ', text_no_whitespace)
    return text_no_punctuation


def q_2(data):
    # a
    data_token = nltk.word_tokenize(data)
    stemmer = PorterStemmer()
    word_stemmed = [stemmer.stem(val).lower() for val in data_token]
    print('word_stemmed=', word_stemmed)
    # b
    data_token = nltk.word_tokenize(data)
    data_normalisation = [word.lower() for word in data_token]
    postagged = nltk.pos_tag(data_normalisation)
    word_lemmatized = [nltk.WordNetLemmatizer().lemmatize(postagged[idx][0], get_wordnet_pos(postagged[idx][1])) for idx
                       in range(len(postagged))]
    print('word_lemmatized=', word_lemmatized)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def q_3(url):
    raw_html = urllib.request.urlopen(url).read()
    soup = bs4.BeautifulSoup(raw_html, features='lxml')
    title = soup.title
    print('title:', title)
    summary = soup.summary
    print('summary:', summary)
    body = soup.body.get_text(strip=True)
    print('body:', body)


if __name__ == '__main__':
    data = open('text').read()
    url = 'https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html'
    q_1(data)
    q_2(data)
    q_3(url)
