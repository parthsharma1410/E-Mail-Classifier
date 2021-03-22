#importing nltk library
import nltk
nltk.download('punkt')

#collecting training data
import csv
data = []
with open('mails.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append((row[0].strip(), row[1].strip()))
        
#tokenising
from nltk.tokenize import sent_tokenize, word_tokenize
def tokenizer(sentence):
    tokenised = word_tokenize(sentence)
    return tokenised

#stopword removal
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
def remove_stopwords(wordlist):
    wordsFiltered = []
    for w in wordlist:
        if w not in stopWords:
            wordsFiltered.append(w)

    return wordsFiltered

#checking part of speech
verb_pos = []
def pos(filtered_words):
    tagged = nltk.pos_tag(filtered_words)
    for word,pos in tagged:
        if pos == 'VBD' or 'VBG' or 'VBN' or 'VBP' or 'VBZ':
            verb_pos.append(word)
    return verb_pos

#stemming
from nltk.stem import PorterStemmer
result = []
def my_stemmer(wordlist):
    ps = PorterStemmer()
    for word in wordlist:
        stemword = ps.stem(word)
        result.append(stemword)
    return result

#extract unique words
unique_words = set(word.lower().strip('.') for review in data for word in my_stemmer(pos(remove_stopwords(tokenizer(review[0])))))

#create a feature set
train_data = [({word: (word in my_stemmer(remove_stopwords(tokenizer(x[0])))) for word in unique_words}, x[1]) for x in data]

#justifying a suitable classifier for the dataset based on accuracy and other parameters 
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)
test_sentence = 'Get the opportunity to win free prizes'
test_sentence_features = {word: (word in my_stemmer(pos(remove_stopwords(tokenizer(test_sentence))))) for word in test_sentence}
classifier.classify(test_sentence_features)