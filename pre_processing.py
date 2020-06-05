import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from statistics import stdev
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import re
# from unidecode import unidecode
import unicodedata
import csv


def pre_processing(df):
	# remove starting and ending double quotes (") 
	df['Comment'] = df['Comment'].str.strip('"')

	# convert all characters to lowercase
	df['Comment'] = df['Comment'].str.lower()

	# remove all URLs
	df['Comment'] = df['Comment'].str.replace(r'https?:\/\/.*[\r\n]*', ' ')

	# remove all escaping characters
	# e.g. 'a\\xc2\\xa0majority of canadians can' is converted to
	# 'a majority of canadians can'
	new_rows = []
	for row in df['Comment']:
		new_row = row

		# e.g. 'a\\xc2\\xa0majority of canadians can' is converted to
		# 'a\xc2\xa0majority of canadians can'
		new_row = new_row.replace('\\\\', '\\')

		# e.g. 'a\xc2\xa0majority of canadians can' is converted to
		# 'a majority of canadians can'
		new_row = new_row.encode().decode('unicode_escape')

		new_rows.append(new_row)
	df['Comment'] = new_rows


	# remove all specials characters ('.', ',', '\n' ...) except apostrophes '\''
	df['Comment'] = df['Comment'].str.replace('[^a-z\']+', ' ')

	return df

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

def lemmas(data):
	lemmatizer = WordNetLemmatizer() 
	return [lemmatizer.lemmatize(token, pos=wordnet_pos_code(tag)) for token, tag in data]

def _lemmatization(data):
	try:
		data = lemmas(data)
	except LookupError:
		nltk.download('punkt')
		nltk.download('wordnet')
		nltk.download('averaged_perceptron_tagger')
		data = lemmas(data)
	return ' '.join(data)

def lemmatization(docs):
	new_docs = []
	for doc in docs:
		doc = pos_tag(word_tokenize(doc))
		new_docs.append(_lemmatization(doc))

	return new_docs

def run_naive_bayes(X_train, y_train, X_test, y_test, _alpha=0.5):
	# fit
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB(alpha=_alpha)
	clf.fit(X_train, y_train)

	# predict
	predictions_count = clf.predict(X_test)

	# print evaluation score
	print("f1 score: ", f1_score(y_test, predictions_count))
	# np.set_printoptions(threshold=sys.maxsize)
	# print(predictions_count)
	# print(y_test.tolist())

if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train.csv", sep=',')
	df_test = pd.read_csv("../data/impermium_verification_labels.csv", sep=',')

	# pre processing
	df_train = pre_processing(df_train)
	df_test = pre_processing(df_test)


	# with open("../test.csv", 'w+') as file:
	# 	file.write(df_train.to_csv(index=False, sep=','))

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']
	

	#################################### Naive Bayes ####################################
	######### naive bayes without any optimizations #########
	count_vectorizer = CountVectorizer()
	X_train_count = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_count = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes :")
	run_naive_bayes(X_train_count.toarray(), y_train,
			X_test_count.toarray(), y_test)


	######### Naive Bayes with optimizations #########
	######### lemmatization #########
	# copy is needed because each optimization is independent
	df_train_lemma = df_train.copy()
	df_test_lemma = df_test.copy()

	df_train_lemma['Comment'] = lemmatization(df_train_lemma['Comment'].tolist())
	df_test_lemma['Comment'] = lemmatization(df_test_lemma['Comment'].tolist())

	count_vectorizer = CountVectorizer()
	X_train_count = count_vectorizer.fit_transform(df_train_lemma['Comment'])
	X_test_count = count_vectorizer.transform(df_test_lemma['Comment'])

	print("Naive Bayes after lemmatization :")
	run_naive_bayes(X_train_count.toarray(), y_train,
			X_test_count.toarray(), y_test)

	######### stop words #########
	count_vectorizer = CountVectorizer(stop_words='english')
	X_train_count = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_count = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes after removing stop-words :")
	run_naive_bayes(X_train_count.toarray(), y_train,
			X_test_count.toarray(), y_test)

	######### n-grams #########
	count_vectorizer = CountVectorizer(ngram_range=(2, 2))
	X_train_count = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_count = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes with bigrams :")
	run_naive_bayes(X_train_count.toarray(), y_train,
			X_test_count.toarray(), y_test)

	######### laplace smoothing #########
	count_vectorizer = CountVectorizer()
	X_train_count = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_count = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes with laplace smoothing :")
	run_naive_bayes(X_train_count.toarray(), y_train,
			X_test_count.toarray(), y_test, 1)
