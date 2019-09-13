import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score as acs
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from helper import count_punctuation, remove_punctuation, tokenize, remove_stopwords, stemming, lemmatizing

# nltk.download()
# raw_data = open('SMSSpamCollection.tsv').read()


data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'body_text'], header=None)
data.columns = ['label', 'body_text']

#START DATA PREPROCESSING

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(' '))

data['punct%'] = data['body_text'].apply(lambda x: count_punctuation(x))

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punctuation(x))

data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))

stopwords = nltk.corpus.stopwords.words('english')

data['body_text_nonstop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x, stopwords))

stemmer = nltk.PorterStemmer()

data['body_text_stemmed'] = data['body_text_nonstop'].apply(lambda x: stemming(x, stemmer))

wn = nltk.WordNetLemmatizer()

data['body_text_lemmatized'] = data['body_text_nonstop'].apply(lambda x: lemmatizing(x, wn))

#END DATA PREPROCESSING

#BEGINNING VECTORIZATION OF DATA

clean_text = data['body_text_stemmed']

count_vec = CountVectorizer()
X_counts = count_vec.fit_transform(data['body_text'])



#SPLIT INTO TRAIN TEST DATA

X=data[['body_text', 'body_len', 'punct%']]
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#END SPLITTING TRAIN TEST DATA

#BEGIN VECTORIZING TEXT

tfidf_vect = TfidfVectorizer()
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])

tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
           pd.DataFrame(tfidf_test.toarray())], axis=1)

#END VECTORIZING TEXT

#BEGIN EVALUATING MODELS

rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train_vect, y_train)

y_pred = rf_model.predict(X_test_vect)

precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(y_test,y_pred), 3)))

#END EVALUATING MODELS

#BEGIN MAKING CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)
class_label = ["ham", "spam"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#END MAKING CONFUSION MATRIX
