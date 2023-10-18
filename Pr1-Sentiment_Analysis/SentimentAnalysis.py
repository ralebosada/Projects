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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

## TRAINING NAIVE BAYES MODEL

# Downloading Dataset
# Data set: https://huggingface.co/datasets/rotten_tomatoes
dataset = load_dataset('rotten_tomatoes', split='train')
dataset_test = load_dataset('rotten_tomatoes', split='test')

# Loading Dataset using Pandas
df = pd.DataFrame({'text':dataset['text'], 'label':dataset['label']}, columns=['text', 'label'])

# Extracting Dataset
X_train = df['text']  # features
y_train = df['label'] # target (positive=1, negative=0)

# Created Features using SKLearn
my_vectorizer = TfidfVectorizer()  # Removed Stop words and used TfidVectorizer()
X_train_vec = my_vectorizer.fit_transform(X_train)

# Created Naive-Bayes Model using SKLearn
my_model = MultinomialNB()
my_model.fit(X_train_vec, y_train)
y_pred = my_model.predict(X_train_vec) # Note: this should be done a separate test set
y_true = y_train                    # Note: this should be done a separate test set

# Print some of the prediction results
prediction_results = pd.DataFrame({'Target':y_true, 'Prediction':y_pred}, columns=['Target', 'Prediction'])

# Plot Confusion Matrix for Training
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Prediction')
ax.set_ylabel('Target')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['positive', 'negative'])
ax.yaxis.set_ticklabels(['positive', 'negative'])

plt.show()

# Print performance statistics
print("\n",classification_report(y_true, y_pred, labels=[1, 0], target_names=['Positive', 'Negative']))

## TESTING NAIVE-BAYES MODEL

# Loading Test Dataset
df_test = pd.DataFrame({'text':dataset_test['text'], 'label':dataset_test['label']}, columns=['text', 'label'])
X_test = df_test['text']
y_test = df_test['label']

# Created Features using SKLearn
X_test_vec = my_vectorizer.transform(X_test)
counts_test = X_test_vec.toarray().sum(axis=0)

# Evaluating NB Model using Test Dataset
y_pred, y_true = [0]*10, [1]*10
y_pred = my_model.predict(X_test_vec)
y_true = y_test

# Plot Confusion Matrix for Testing
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
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

# References:
# PRRegonia:CS280_Lesson_1