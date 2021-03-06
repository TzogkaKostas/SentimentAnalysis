import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# WordNetLemmatizer 'understands' only 4 tags. Also, every word not
# tagged as 'ADJ', 'ADV' or 'VERB' (VBD, VBG, VBN, ...) is consired as a
# NOUN. In order to avoid this, every tag is converted (wordnet_pos_code)
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

	# Lemmatize each word depending on its tag. 
	return [lemmatizer.lemmatize(token, pos=wordnet_pos_code(tag)) 
			for token, tag in data]

def _lemmatization(data):
	try:
		data = lemmas(data)
	except LookupError:
		nltk.download('punkt')
		nltk.download('wordnet')
		nltk.download('averaged_perceptron_tagger')
		data = lemmas(data)
	return ' '.join(data)

def lemmatization(comments):
	new_comments = []
	for comment in comments:
		# tokenize the comment and find POS tag for each word
		comment_with_tags = pos_tag(word_tokenize(comment))
		new_comments.append(_lemmatization(comment_with_tags))

	return new_comments

def run_naive_bayes(clf, X_train, y_train, X_test, y_test):
	clf.fit(X_train, y_train)

	predictions_count = clf.predict(X_test)
	print("f1 score: ", f1_score(y_test, predictions_count))
	print("accuracy score: ", accuracy_score(y_test, predictions_count))

def run_multinomial_nb(X_train, y_train, X_test, y_test, _alpha=1):
	clf = MultinomialNB(alpha=_alpha)
	run_naive_bayes(clf, X_train, y_train, X_test, y_test)

def run_gaussian_nb(X_train, y_train, X_test, y_test):
	clf = GaussianNB()
	run_naive_bayes(clf, X_train, y_train, X_test, y_test)

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

	print("Naive Bayes without any optimizations :")
	run_gaussian_nb(X_train.toarray(), y_train, X_test.toarray(), y_test)

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
	run_gaussian_nb(X_train_lemma.toarray(), y_train, X_test_lemma.toarray(),
			y_test)

	######### stop words #########
	count_vectorizer = CountVectorizer(stop_words='english')
	X_train_stop = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_stop = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes after removing stop-words :")
	run_gaussian_nb(X_train_stop.toarray(), y_train, X_test_stop.toarray(),
			y_test)

	######### n-grams #########
	count_vectorizer = CountVectorizer(ngram_range=(2, 2))
	X_train_gram = count_vectorizer.fit_transform(df_train['Comment'])
	X_test_gram = count_vectorizer.transform(df_test['Comment'])

	print("Naive Bayes with bigrams :")
	run_gaussian_nb(X_train_gram.toarray(), y_train,
			X_test_gram.toarray(), y_test)

	######### laplace smoothing #########
	print("Naive Bayes with laplace smoothing :")
	run_multinomial_nb(X_train.toarray(), y_train,
			X_test.toarray(), y_test)