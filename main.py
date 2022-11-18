
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#cross val function returning f1, accuracy, precision, and recall performance of classifier parameter
def crossVal(classifier):
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=10)
    print('F1: ' + str(round(100 * f1.mean(), 2)) + "%")
    f1 = cross_val_score(classifier, X, y, scoring='accuracy', cv=10)
    print('Accuracy: ' + str(round(100 * f1.mean(), 2)) + "%")
    f1 = cross_val_score(classifier, X, y, scoring='precision', cv=10)
    print('Precision: ' + str(round(100 * f1.mean(), 2)) + "%")
    f1 = cross_val_score(classifier, X, y, scoring='recall', cv=10)
    print("Recall: " + str(round(100 * f1.mean(), 2)) + "%")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #series of arrays to store the organized comments for later classification
    path = 'icpc17'
    concernfiles = []
    miscfiles = [] #others comments
    allcomments = [] #combines all comments

    # loop through files to extract concerns and others txt files
    for file in os.listdir(path):
        if file != '.DS_Store':  # or file != 'ReadME.txt':
            with open(path + '/' + file + '/concerns.txt') as fil:
                for line in fil.readlines():
                    concernfiles.append(line) #append each concern line to a concern file
                    allcomments.append(line) #append each line to an allcomments file

            with open(path + '/' + file + '/others.txt') as fil:
                for line in fil.readlines():
                    miscfiles.append(line) #append each concern line to a miscellanous file
                    allcomments.append(line) #append each line to an allcomments file

    zeros = np.zeros(len(concernfiles)) #fill each line in concern file as a 0 representation
    ones = np.ones(len(miscfiles)) #fill each line in others file as a 0 representation

    y = np.append(zeros, ones) #prepare for y value for classifers

    # vectorizer to count frequency of words in commont
    vectorizer = CountVectorizer()

    #count frequency of words in all comments
    X = vectorizer.fit_transform(allcomments)
    #print("\nDimensions of training data:", X.shape)

    # Create the tf-idf transformer
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X)
    #print("\nDimensions of tfidf training data:", X_tfidf.shape)

    #initailizes all classifiers: SVM, Naive Bayes, Decision Tree, Random Forest Tree
    svmclassifier = LinearSVC(random_state=0)

    nbclassifier = MultinomialNB()

    params = {'random_state': 0, 'max_depth': 4}
    dtclassifier = DecisionTreeClassifier(**params)

    rfclassifier = RandomForestClassifier(**params)

    print('SVM Classifier')
    crossVal(svmclassifier)

    print('Naive Bayes Classifier')
    crossVal(nbclassifier)

    print('Decision Tree Classifier')
    crossVal(dtclassifier)

    print('Random Forest Classifier')
    crossVal(rfclassifier)


