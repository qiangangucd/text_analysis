import nltk
from nltk.corpus import names
import random
from nltk.classify import accuracy
import matplotlib.pyplot as plt


def gender_features(word):
    # return {'last_letter': word[-1]}
    # gender_features('Shrek') = {'last_letter': 'k'}
    return {'last_letter': word[-1]}, \
           {'last_two_letter':word[-2:]},\
           {'last_three_letter':word[-3:]},\
           {'length_name':len(word)},\
           {'fitst_letter':word[0]},\
           {'first_two_letter':word[:2]},\
           {'last_letter': word[-1], 'last_two_letters': word[-2:]}, \
           {'last_letter': word[-1], 'last_two_letters': word[-2:], 'last_three_letters': word[-3:]}, \
           {'last_letter': word[-1], 'last_two_letters': word[-2:], 'last_three_letters': word[-3:], 'length_name': len(word)},\
           {'last_letter': word[-1], 'last_two_letters': word[-2:], 'last_three_letters': word[-3:], 'length_name': len(word),'first_letter':word[0]},\
           {'last_letter': word[-1], 'last_two_letters': word[-2:], 'last_three_letters': word[-3:], 'length_name': len(word),'first_letter':word[0],'first_two_letter':word[:2]},\
           {'last_letter': word[-1], 'last_two_letters': word[-2:], 'length_name': len(word),'first_letter':word[0],'first_two_letter':word[:2]}


# 'last_two_letters': word[-2:], 'length_name': len(word)


male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)
# featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
# # entries are    ({'last_letter': 'g'}, 'male')
# train_set, test_set = featuresets[500:], featuresets[:500]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
#
# ans1 = classifier.classify(gender_features('Mark'))
# ans2 = classifier.classify(gender_features('Precilla'))
#
# print("Mark is:", ans1)
# print("Precilla is:", ans2)
# print(accuracy(classifier, test_set))
# classifier.show_most_informative_features(5)
# print(nltk.classify.accuracy(classifier, test_set))
acc=[]
for i in range(12):
    featuresets = [(gender_features(n)[i], gender) for (n, gender) in labeled_names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc.append(accuracy(classifier, test_set))

print(acc)
plt.plot(range(12),acc)
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.xticks(range(12))
plt.savefig('bc.png',dpi=100,bbox_inches='tight')
plt.show()

