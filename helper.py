from string import punctuation
import re

def count_punctuation(text):
    count = 0
    for char in text:
        if char in punctuation:
            count += 1
    
    return round(count/(len(text) - text.count(" ")), 3)*100

def remove_punctuation(text):
    text = ''.join([char for char in text if char not in punctuation])
    return text

def tokenize(text):
    tokens = re.split('\W+', text) #W+ means a word character (A-Z a-z 0-9) or a dash can go there
    return tokens

def remove_stopwords(tokenized_list, stopwords):
    text = [word for word in tokenized_list if word not in stopwords]
    return text

def stemming(tokenized_text, stemmer):
    text = [stemmer.stem(word) for word in tokenized_text]
    return text

def lemmatizing(tokenized_text, wn):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text