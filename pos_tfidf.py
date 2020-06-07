import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize 
from collections import Counter
import numpy as np

def wordnet_pos_code(tag):
	if tag in ['JJ', 'JJR', 'JJS']:
		return wn.ADJ
	elif tag in ['RB', 'RBR', 'RBS']:
		return wn.ADV
	elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
		return wn.NOUN
	elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		return wn.VERB
	return wn.NOUN

def _pos_fractions(comment_with_tags):
	size = len(comment_with_tags)

	# empty comment
	if size == 0 :
		return [0, 0, 0, 0]

	count_adverbs = 0
	count_verbs = 0
	count_adjectives = 0
	count_nouns = 0
	for word, tag in comment_with_tags:
		if wn.ADV == wordnet_pos_code(tag):
			count_adverbs += 1
		if wn.VERB == wordnet_pos_code(tag):
			count_verbs += 1
		if wn.ADJ == wordnet_pos_code(tag):
			count_adjectives += 1
		if wn.NOUN == wordnet_pos_code(tag):
			count_nouns += 1

	return [count_adverbs/size, count_verbs/size, count_adjectives/size, 
			count_nouns/size]

def pos_fractions(comments):
	fractions_array = []
	for comment in comments:
		comment_with_tags = pos_tag(word_tokenize(comment))
		fractions = _pos_fractions(comment_with_tags)
		fractions_array.append(fractions)

	return fractions_array


if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_cleaned.csv", sep=',')
	df_test = pd.read_csv("../data/test_cleaned.csv", sep=',')

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']

	# text vectorization by TfidfVectorizer
	tfidf_vectorizer = TfidfVectorizer()
	X_train = tfidf_vectorizer.fit_transform(df_train['Comment'])
	X_test = tfidf_vectorizer.transform(df_test['Comment'])

	# Part-of-Speech fractions
	fractions_array_train = pos_fractions(df_train['Comment'].tolist())
	fractions_array_test = pos_fractions(df_test['Comment'].tolist())

	# combine the 2 matrices (append the 4 fractions for each vector)
	X_train = np.hstack((X_train.toarray(), fractions_array_train))
	X_test = np.hstack((X_test.toarray(), fractions_array_test))


	#################################### SVM ####################################
	# clf = svm.SVC(kernel='linear')
	# # clf = GridSearchCV(clf, {'kernel':('linear', 'rbf'), 'C':[1, 10]}, n_jobs=-1)
	# clf.fit(X_train, y_train)

	# predictions = clf.predict(X_test)

	# print("f1 score: ", f1_score(y_test, predictions))
	# print("accuracy score: ", accuracy_score(y_test, predictions))

	#################################### Random Forrest ####################################
	clf = RandomForestClassifier()
	# clf = GridSearchCV(clf, n_jobs=4, param_grid=
			# {'max_depth':[None,1,2,4,8,16,32,64], ''})
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)

	print("f1 score: ", f1_score(y_test, predictions))
	print("accuracy score: ", accuracy_score(y_test, predictions))