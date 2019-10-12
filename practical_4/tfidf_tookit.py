import sys
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from wordcloud import WordCloud

# stemmer = PorterStemmer() # Stemmer for reducing terms to root form
#
# stemmed_corpus = []       # For storing the stemmed tokens
#
# original_corpus = []      # For storing the non-stemmed tokens


# for file in sys.argv[1:]:                # Iterate over the files
#
#     contents = open(file).read().lower() # Load file contents
#
#     tokens = word_tokenize(contents)     # Extract tokens
#
#     stemmed = [stemmer.stem(token) for token in tokens] # Stem tokens
#
#
#     stemmed_corpus.append(stemmed)    # Store stemmed document
#
#     original_corpus.append(tokens)    # Store original document

stemmed_corpus = [['today', 'go', 'nlp', 'course', 'learn', 'segment', 'word', 'extract', 'stem', 'preprocess', 'text'],
                  ['today', 'go', 'school', 'want', 'participant', 'text', 'analyse', 'course', 'sound', 'nice', 'due',
                   'segment', 'word'],
                  ['yesterday', 'go', 'data', 'science', 'course', 'learnt', 'preprocess', 'text', 'happy', 'due',
                   'teacher', 'nice'],
                  ['today', 'nlp', 'course', 'interesting', 'learnt', 'lot', 'knowledge', 'useful', 'career', 'happy'],
                  ['today', 'go', 'school', 'want', 'participant', 'data', 'mining', 'course', 'sound', 'nice', 'due',
                   'secret'],
                  ['machine', 'learn', 'course', 'also', 'nice', 'teacher', 'pleased', 'professional', 'teach', 'u',
                   'useful', 'technology'],
                  ['weather', 'terrible', 'result', 'missed', 'bus', 'late', 'school', 'result', 'miss', 'text',
                   'analyse', 'course'],
                  ['teacher', 'taught', 'u', 'process', 'raw', 'data', 'write', 'code', 'implement', 'algorithm',
                   'evaluate', 'machine', 'learning', 'algorithm'],
                  ['today', 'weather', 'nice', 'go', 'school', 'absence', 'machine', 'learn', 'course', 'climb',
                   'mountain', 'picnic'],
                  ['today', 'go', 'gym', 'sport', 'finish', 'homework', 'assign', 'machine', 'data', 'mining', 'course',
                   'feel', 'well', 'exercise']]
dictionary = Dictionary(stemmed_corpus)  # Build the dictionary

# Convert to vector corpus

vectors = [dictionary.doc2bow(text) for text in stemmed_corpus]

# Build TF-IDF model

tfidf = TfidfModel(vectors)

# Get TF-IDF weights
weights = tfidf[vectors[0]]
# Get terms from the dictionary and pair with weights

weights = dict((dictionary[pair[0]], pair[1]) for pair in weights)

# Initialize the word cloud

wc = WordCloud(
    background_color="white",
    # max_words=2000,
    width=500,
    height=500,
    stopwords=stopwords.words("english")
)

# Generate the cloud

wc.generate_from_frequencies(weights)

# Save the could to a file

wc.to_file("word_cloud.png")
