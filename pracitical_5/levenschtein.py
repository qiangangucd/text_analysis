__author__ = 'user'

from nltk.metrics import edit_distance
import distance
import matplotlib.pyplot as plt

#  transposition flag allows transpositions edits (e.g., “ab” -> “ba”),
#
# s1 = 'dr mark keane'
# s2 = 'mr mark bean'
#
# s3 = 'rain'
# s4 = 'shine'
#
# s5 = 'rowan mr atkinson'
# s6 = 'mr bean'
# ans = edit_distance(s1, s2, transpositions=False)
# print(ans)
# #
# ans = edit_distance(s3, s4, transpositions=False)
# print(ans)
# #
# ans = edit_distance(s5, s6, transpositions=False)
# print(ans)
# #
# ans = distance.levenshtein(s1, s2)
# print(ans)
# #
# ans = distance.levenshtein(s3, s4)
# print(ans)
# #
# ans = distance.levenshtein(s5, s6)
# print(ans)
s1 = 'We are delighted to let you know that over 1,700 people have registered for the conference.'
s2 = 'It’s heartening that I just started teaching by another school.'
s3 = 'Attitude is a choice. Happiness is a choice. Optimism is a choice. Kindness is a choice.'
s4 = 'However batty my parents sometimes drive me.'
s5 = 'This is a pretty devastating moment for the DUP'
s1_spam='We are delighted to let you know that over 1,700 people have registered for the conference. All for free'
s2_spam = '#Cash4Ash We are delighted to let you know that over 1,700 people have registered for the conference.'
s3_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. #Cash4Ash'
s4_spam = ' http://stackoverflow.com/questions/33398282/attributeerror-module-object-has-no-attribute-scores We are delighted to let you know that over 1,700 people have registered for the conference.'
s5_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference.  http://stackoverflow.com/questions/33398282/attributeerror-module-object-has-no-attribute-scores'
s6_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. Find out more: https://goo.gl/A3Ctb9'
s7_spam = ' FREE Smart TV, bundles from €20. We are delighted to let you know that over 1,700 people have registered for the conference.'
s8_spam = 'Find out more: https://goo.gl/A3Ctb9 We are delighted to let you know that over 1,700 people have registered for the conference.'
s9_spam = 'FREE Smart TV, bundles from €20. We are delighted to let you know that over 1,700 people have registered for the conference.'
s10_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. Offer ends tomorrow'
s11_spam = 'Offer ends tomorrow. We are delighted to let you know that over 1,700 people have registered for the conference. '
s12_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. all for FREE'
s13_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. all for FREE'
s14_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. #plasticpollution #plasticwaste #plasticfree'
s15_spam = '#plasticpollution #plasticwaste #plasticfree. We are delighted to let you know that over 1,700 people have registered for the conference.'
s16_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference. #becauseideserveit #youknowitmakessense #ifyouarethishandsomeitcomesnaturally'
s17_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference.#becauseideserveit #youknowitmakessense'
s18_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference.#becauseideserveit #ifyouarethishandsomeitcomesnaturally'
s19_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference.#becauseideserveit'
s20_spam = 'We are delighted to let you know that over 1,700 people have registered for the conference.#becauseideserveit for free'
s_norm=[s1,s2,s3,s4,s5]
s_spam=[s1_spam,s2_spam,s3_spam,s4_spam,s5_spam,s6_spam,s7_spam,s8_spam,s9_spam,s10_spam,s11_spam,s12_spam,s13_spam,s14_spam,s15_spam,s16_spam,s17_spam,s18_spam,s19_spam,s20_spam]
ed_norm=[]
for i in range(len(s_norm)):
    for j in range(i+1,len(s_norm)):
        ed_norm.append(distance.levenshtein(s_norm[i],s_norm[j]))
ed_spam=[]
for i in range(len(s_spam)):
    for j in range(i+1,len(s_spam)):
        ed_spam.append(distance.levenshtein(s_spam[i],s_spam[j]))


plt.scatter(range(0,190,19),ed_norm,color='g',label='Norm')
plt.scatter(range(len(ed_spam)),ed_spam,color='r',label='Spam')
plt.legend()
plt.savefig('norm_spam.png',dpi=100,bbox_inches='tight')
plt.show()


