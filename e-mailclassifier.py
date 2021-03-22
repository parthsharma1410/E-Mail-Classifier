#This program detects if an email is spam or not

#collecting training data
import csv
data = []
with open('mails.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append((row[0].strip(), row[1].strip()))
data

#preprocessing training data
def tokenizer(sentence):
    return sentence.split(" ")
    
#removing stopwords
def remove_stopwords(wordlist):
    stopwords = ['i', 'me', 'my', 'myself', 'our', 'ours', 'ourselves','it','is', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'while', 'of', 'at',
                 'by', 'any' ,'for', 'with', 'about', 'between', 'into',"shouldn't", "wasn't", "weren't", 
                 "won't", "wouldn't"]
    return [w for w in wordlist if w not in stopwords]
    
#making a stemmer dictionary and checking if word is in stemmer dictionary
def my_stemmer(wordlist):
    stemmer_dict = {
        'prize' : ('prize', 'prizes', 'prized'),
        'win' : ('win','won', 'winning', 'wins'),
        'opportunity': ('opportunity','opportunities'),
        'selection': ('selection', 'selecting', 'selected'),
        'free': ('free'),
        'challenge': ('challenge', 'challenges', 'challenged'),
        'important': ('important', 'importance'),
        'education': ('education', 'educated'),
        'technology': ('technology', 'technologies')
    }
    result = []
    for word in wordlist:
        for key, value in stemmer_dict.items():
            if word in value:
                result.append(key)
    return result
    
#extract unique words
unique_words = set(word.lower().strip('.') for review in data for word in my_stemmer(remove_stopwords(tokenizer(review[0]))))

#create a feature set
train_data = [({word: (word in my_stemmer(remove_stopwords(tokenizer(x[0])))) for word in unique_words}, x[1]) for x in data]

#justifying a suitable classifier for the dataset based on accuracy and other parameters 
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)
test_sentence = 'Get opportunity to win free prizes'
test_sentence_features = {word: (word in my_stemmer(remove_stopwords(tokenizer(test_sentence)))) for word in test_sentence}
classifier.classify(test_sentence_features)
