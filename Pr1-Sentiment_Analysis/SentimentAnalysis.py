'''
CS 280 - Machine Problem 1
Instructions: 
1.) Download requirements.txt
2.) Run NBClassifier.py

Thank you!
'''

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r" ,package])

install('requirements.txt')

from datasets import load_dataset
from pprint import pprint

dataset = load_dataset('rotten_tomatoes', split='train')
print(dataset.info.description)

import numpy as np 
import pandas as pd 

df = pd.DataFrame({'text':dataset['text'], 'label':dataset['label']}, columns=['text', 'label'])
X_train = df['text']  # features
y_train = df['label'] # target (positive=1, negative=0)

print("Sample positive reviews")
print(df[df['label']==1][:5])

print("\nSample negative reviews")
print(df[df['label']==0][:5])

from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer()
vectorizer = CountVectorizer(stop_words='english')  # remove stop words
X_train_vec = vectorizer.fit_transform(X_train)

words = vectorizer.get_feature_names_out()
counts = X_train_vec.toarray().sum(axis=0)

df_words = pd.DataFrame({'words':words, 'count':counts}, columns=['words', 'count'])
df_words.sort_values(by=['count'], ascending=False).head(10)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_train_vec) # Note: this should be done a separate test set
y_true = y_train                    # Note: this should be done a separate test set

# Print some of the prediction results
prediction_results = pd.DataFrame({'Target':y_true, 'Prediction':y_pred}, columns=['Target', 'Prediction'])
print(prediction_results.sample(n=20, random_state=280))

# Compute confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

# Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Prediction')
ax.set_ylabel('Target')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['positive', 'negative'])
ax.yaxis.set_ticklabels(['positive', 'negative'])

plt.show()

# Print performance statistics
from sklearn.metrics import classification_report

print("\n",classification_report(y_true, y_pred, labels=[1, 0], target_names=['Positive', 'Negative']))