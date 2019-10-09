import pandas as pd
import scipy.stats
import nltk
import string
import pandas as pd
import numpy as np
from math import log, e
from scipy.stats import entropy
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import math
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


def get_tokens(text):
    lowers = text.lower()
    # remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    postagged = nltk.pos_tag(filtered)
    word_lemmatized = [nltk.WordNetLemmatizer().lemmatize(postagged[idx][0], get_wordnet_pos(postagged[idx][1])) for idx
                       in range(len(postagged))]
    count = nltk.Counter(word_lemmatized)
    return word_lemmatized, count


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


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def compute_tfidf(countlist):
    for i, count in enumerate(countlist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, count, countlist) for word in count}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        df = round(pd.DataFrame([scores]), 5)
        df.to_csv('df' + str(i) + '.csv', index=False)
        for word, score in sorted_words:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


def compute_tf(countlist):
    for i, count in enumerate(countlist):
        scores = {word: tf(word, count) for word in count}
        df = round(pd.DataFrame([scores]), 5)
        # df.to_csv('df'+str(i)+'.csv',index=False)
        print(round(df, 5))


# def entropy1(labels, base=None):
#     value, counts = np.unique(labels, return_counts=True)
#     return entropy(counts, base=base)
#
#
# def entropy3(labels, base=None):
#     vc = pd.Series(labels).value_counts(normalize=True, sort=False)
#     base = e if base is None else base
#     return -(vc * np.log(vc) / np.log(base)).sum()
#
#
# def entropy4(labels, base=None):
#     value, counts = np.unique(labels, return_counts=True)
#     norm_counts = counts / counts.sum()
#     base = e if base is None else base
#     return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()  # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


if __name__ == '__main__':
    text0 = 'Today we went to the NLP course and learned how to segment words, how to extract stems and preprocess ' \
            'the text. '
    text1 = 'today they are going to school and want to participant the text analyse course. It sounds very' \
            ' nice due to the segment words'
    text2 = 'Yesterday they went to a data science course and learnt how to preprocess the text. they are very happy ' \
            'due to the teacher was very nice '
    text3 = 'today\'s nlp course is very interesting, we have learnt a lot of knowledge and it is very useful for our ' \
            'careers, so we are very happy '
    text4 = 'today they are going to school and want to participant the data mining course. It sounds very' \
            ' nice due to the secret'
    text5 = 'the machine learning course is also very nice, because the teacher is very pleased and professional, ' \
            'he teaches us some useful technology. '
    text6 = 'the weather is very terrible, resulting in he missed his bus and was late for school. As a result, he ' \
            'missed the text analyse course. '
    text7 = 'The teacher taught us how to process the raw data and how to write the code to implement the algorithm ' \
            'and how to evaluate the machine learning algorithm '
    text8 = 'today\'s weather is very nice, so they did not go to school and absence of machine learning course. They ' \
            'just climbed the mountain and have a picnic '
    text9 = 'today they went to the gym and do some sports after they finish their homework assigned in machine data ' \
            'mining course. They feel better through exercising. '
    word_tokens0 = get_tokens(text0)
    word_tokens1 = get_tokens(text1)
    word_tokens2 = get_tokens(text2)
    word_tokens3 = get_tokens(text3)
    word_tokens4 = get_tokens(text4)
    word_tokens5 = get_tokens(text5)
    word_tokens6 = get_tokens(text6)
    word_tokens7 = get_tokens(text7)
    word_tokens8 = get_tokens(text8)
    word_tokens9 = get_tokens(text9)
    # countlist_count = [word_tokens0[1], word_tokens1[1], word_tokens2[1], word_tokens3[1], word_tokens4[1], word_tokens5[1],
    #              word_tokens6[1],
    #              word_tokens7[1], word_tokens8[1], word_tokens9[1]]
    countlist_word = [word_tokens0[0], word_tokens1[0], word_tokens2[0], word_tokens3[0], word_tokens4[0],
                      word_tokens5[0],
                      word_tokens6[0],
                      word_tokens7[0], word_tokens8[0], word_tokens9[0]]
    print(countlist_word)
    # compute_tfidf(countlist)
    # print(word_tokens0[1])
    # print(word_tokens0[1].elements())
    # print(len(word_tokens0[0]))
    # compute_tf(countlist)
    # print('text3=',get_tokens(text3)[0])
    # print('text4=',get_tokens(text4)[0])
    # print('text5=',get_tokens(text5)[0])
    # print('text6=',get_tokens(text6)[0])
    # print('text7=',get_tokens(text7)[0])
    # print('text8=',get_tokens(text8)[0])
    # print('text9=',get_tokens(text9)[0])

    # text = "this is a foo bar bar black sheep  foo bar bar black sheep foo bar bar black sheep shep bar bar black sentence"
    # text='today go nlp course learn segment word extract stem preprocess text today go school want participant text ' \
    #      'analyse course sound nice due segment word yesterday go data science course learnt preprocess text happy ' \
    #      'due teacher nice today nlp course interesting learnt lot knowledge useful career happy today go school want ' \
    #      'participant data mining course sound nice due secret machine learn course also nice teacher pleased ' \
    #      'professional teach u useful technology weather terrible result missed bus late school result miss text ' \
    #      'analyse course teacher taught u process raw data write code implement algorithm evaluate machine learning ' \
    #      'algorithm today weather nice go school absence machine learn course climb mountain picnic today go gym ' \
    #      'sport finish homework assign machine data mining course feel well exercise '
    # bigram_measures = nltk.collocations.QuadgramAssocMeasures()
    # finder = QuadgramCollocationFinder.from_words(word_tokenize(text))
    # finder = TrigramCollocationFinder.from_words(countlist_word)

    # for i in finder.score_ngrams(bigram_measures.pmi):
    #     print(i)

    # print(finder.score_ngrams(bigram_measures.pmi)[0:10])
    spam_set = []
    # spam_set.append('Hair Growth and soothe your scalp with our Ginger shampoo.')
    # spam_set.append('Meeting your new Hair Growth! Introducing the perfect partner to Ginger Shampoo.')
    # spam_set.append('Blease try the body shop Ginger shampoo my Hair Growth by it.')
    # spam_set.append('I used Ginger shampoo and conditioner and spoiler! it dyed my Hair Growth')
    # spam_set.append('Ginger condoms? Ginger tampons? Ginger bedsheets？ Ginger shampoo？')
    # spam_set.append('I bet all the people who bullied me for being ginger would find it Hair Growth')
    # spam_set.append('New Arrival Andrea Hair Growth Products Ginger Shampoo')
    # spam_set.append('Find your perfect Ginger shampoo with Hair Growth')
    # spam_set.append('This Ginger shampoo allegedly makes great shampoo.')
    # spam_set.append('Thanks to Ginger, our Ginger  Shampoo and Conditioner not only refreshes')
    # spam_set.append('Let the Cameras In - Stop secret heath care negotiations')
    # spam_set.append('Stop secret heath care negotiations. Sign the petition to let the Cameras In!')
    # spam_set.append('Let the Cameras In - Stop secret heath care negotiations')
    # spam_set.append('Sign the petition - Let the Cameras In!')
    # spam_set.append('What are Speaker Pelosi,Sen. Reid and Pres Obama hiding?')
    # spam_set.append('Sign the petition - Let the Cameras In!')
    # spam_set.append('Sign the petition - Let the Cameras In!')
    # spam_set.append('No Secret health care negotiations!')
    # spam_set.append('No more secret health care negotiations! Tranparency now.')
    # spam_set.append('Let the Cameras In - Stop the secret health care negotiations')
    # # --------------------------------------------------------------------------------------------------------------------
    # random_set = []
    # random_set.append('This is a nice diagram by Zhengyan Zhang and @BakserWang that shows how many recent pretrained '
    #                   'language models are connected. The GitHub repo contains a full list of relevant papers')
    # random_set.append(
    #     'Working in partnership with @hotpress to curate a list of extraordinary tracks to inspire you in '
    #     'the lead up to World Mental Health Day. Listen now on Spotify.')
    # random_set.append('A Vardy household source tells me that Colleen\'s message could actually put Rebekah in a '
    #                   'stronger position')
    # random_set.append('Imagining the Vardy household right now, Rebekah shrieking blue murder, Jamie\'s on his ninth '
    #                   'Red Bull of the morning, kids are crying')
    # random_set.append('Leo Varadkar has reneged on a secret deal with Boris Johnson to open the way to a Brexit '
    #                   'compromise, a senior Downing Street source claimed yesterday')
    # random_set.append('It was touch and go but Wales have won it! Full-time Wales 29-17 Fiji')
    # random_set.append('It\'s up to you today to start making healthy choices. Not choices that are just healthy for '
    #                   'your body, but healthy for your mind.')
    # random_set.append('I\'ve already perfect the art of fake smiling.')
    # random_set.append('VIDEO: Police and demonstrators clash outside Ecuador\'s National Assembly building as protests '
    #                   'over a fuel hike introduced by President Lenin Moreno\'s government intensify')
    # random_set.append('How are wind farms installed in the sea?')
    #
    # print(ent(pd.Series(['a', 'a', 'b', 'a'])))
    # print(spam_set)
    # print(random_set)
    # print('Entropy values for spam-set=', ent(pd.Series(spam_set)))
    # print('Entropy values for random-set=', ent(pd.Series(random_set)))
    # print('Entropy values for mixed-set=', ent(pd.Series(random_set + spam_set)))
    # print('Entropy values for [\'a\',\'b\',\'a\',\'c\',\'b\',\'a\']=',ent(pd.Series(['a','b','a','c','b','a'])))
    # print('Entropy values for [\'a\',\'a\',\'a\',\'a\',\'a\',\'a\']=', ent(pd.Series(['a', 'a', 'a', 'a', 'a', 'a'])))
    # print('Entropy values for [\'a\',\'b\',\'c\',\'d\',\'e\',\'f\']=', ent(pd.Series(['a', 'b', 'c', 'd', 'e', 'f'])))
