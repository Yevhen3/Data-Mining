import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

column_names = ['id', 'topic', 'sentiment', 'text']

train_data = pd.read_csv('datasets/twitter_training.csv', header=None, names=column_names, sep=',', engine='python', on_bad_lines='skip')
val_data = pd.read_csv('datasets/twitter_validation.csv', header=None, names=column_names, sep=',', engine='python', on_bad_lines='skip')

print("Training set head:")
print(train_data.head())
print("\nValidation set head:")
print(val_data.head())

train_data['text'] = train_data['text'].str.rstrip(' ,')
val_data['text'] = val_data['text'].str.rstrip(' ,')

train_data.dropna(subset=['text', 'sentiment'], inplace=True)
val_data.dropna(subset=['text', 'sentiment'], inplace=True)

vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data['text'])
X_val = vectorizer.transform(val_data['text'])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_data['sentiment'])

y_pred = model.predict(X_val)

print("\nMetrics:")
print(classification_report(val_data['sentiment'], y_pred))

cm = confusion_matrix(val_data['sentiment'], y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# NLTK version
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()
#
# tweet_example = train_data['text'].iloc[0]
# vader_scores = sid.polarity_scores(tweet_example)
# print(f"\nTweet example: {tweet_example}")
# print(f"VADER result: {vader_scores}")