import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB


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
	clf = MultinomialNB(alpha=_alpha)
	clf.fit(X_train, y_train)

	predictions_count = clf.predict(X_test)

	print("f1 score: ", f1_score(y_test, predictions_count))
	print("accuracy score: ", accuracy_score(y_test, predictions_count))

if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_cleaned.csv", sep=',')
	df_test = pd.read_csv("../data/test_cleaned.csv", sep=',')

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']
	
	#################################### Naive Bayes ####################################
	######### naive bayes without any optimizations #########
	count_vectorizer = CountVectorizer()
	X_train = count_vectorizer.fit_transform(df_train['Comment'])
	X_test = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes :")
	run_naive_bayes(X_train.toarray(), y_train,
			X_test.toarray(), y_test)


	######### Naive Bayes with optimizations #########
	######### lemmatization #########
	# copy is needed because each optimization is independent
	df_train_lemma = df_train.copy()
	df_test_lemma = df_test.copy()

	df_train_lemma['Comment'] = lemmatization(df_train_lemma['Comment'].tolist())
	df_test_lemma['Comment'] = lemmatization(df_test_lemma['Comment'].tolist())

	count_vectorizer_lemma = CountVectorizer()
	X_train_lemma = count_vectorizer.fit_transform(df_train_lemma['Comment'])
	X_test_lemma = count_vectorizer.transform(df_test_lemma['Comment'])

	print("Naive Bayes after lemmatization :")
	run_naive_bayes(X_train_lemma.toarray(), y_train,
			X_test_lemma.toarray(), y_test)

	######### stop words #########
	count_vectorizer = CountVectorizer(stop_words='english')
	X_train_stop = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_stop = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes after removing stop-words :")
	run_naive_bayes(X_train_stop.toarray(), y_train,
			X_test_stop.toarray(), y_test)

	######### n-grams #########
	count_vectorizer = CountVectorizer(ngram_range=(2, 2))
	X_train_gram = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_gram = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes with bigrams :")
	run_naive_bayes(X_train_gram.toarray(), y_train,
			X_test_gram.toarray(), y_test)

	######### laplace smoothing #########
	print("Naive Bayes with laplace smoothing :")
	run_naive_bayes(X_train.toarray(), y_train,
			X_test.toarray(), y_test, 1)

	######### BEAT THE BENCHMARK #########
	count_vectorizer = CountVectorizer(stop_words='english')
	X_train_beat = count_vectorizer.fit_transform(df_train_lemma['Comment'])
	X_test_beat = count_vectorizer.transform(df_test_lemma['Comment'])

	print("All optimazations :")
	run_naive_bayes(X_train_beat.toarray(), y_train,
			X_test_beat.toarray(), y_test, 1)