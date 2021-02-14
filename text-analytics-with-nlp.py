# Text Analytics with NLP Tools


# Import sci-kit learn metrics module for accuracy calculation

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# Performing Sentiment Analysis
import pandas as pd

# Loads data
data = pd.read_csv('train.tsv', sep='\t')
data.head()
data.info()
data.Sentiment.value_counts()

Sentiment_count = data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrases'])
plt.xlabel('Review Sentiments')
plt/ylabel('Number of Review')
plt.show()


token = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Creates Document-Term Matrix(DTM)
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(data['Phrase'])

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)


# Model Generation using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy: ", metrics.accuracy_score(y_test, predicted))

# Classification rate of 60.49 using CountVector(BoW)
# Outputs:
# MultinomialNB Accuracy: 0604916912299


''' Feature Generation using TF-IDF
tf = TfidVectorizer()
text_tf = tf.fit_transform(data['Phrase'])
X_train, X_test, y_train, y_test = train_test_split(
                                                    text_tf, data['Sentiment'], test_size=0.3, random_state=124)
                                                    
''' Model Generation '''
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy: ", metrics.accuracy_score(y_test, predicted))

# Outputs:
# MultinomialNB Accuracy: 0.586526549618
