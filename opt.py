import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
	# WordNetLemmatizer 'understands' only 4 tags. Also, every word not
	# tagged as 'ADJ', 'ADV' or 'VERB' (VBD, VBG, VBN, ...) is consired as a NOUN. 
	# In order to avoid this, every tag is converted (wordnet_pos_code)
	# Lemmatize each word depending on its tag. 
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

def lemmatization(comments):
	new_docs = []
	for comment in comments:
		# tokenize the comment and find POS tag for each word
		try:
			comment_with_tags = pos_tag(word_tokenize(comment))
			new_docs.append(_lemmatization(comment_with_tags))
		except: # in case of empty comment
			new_docs.append('')

	return new_docs

def run_naive_bayes(X_train, y_train, X_test, y_test, _alpha=0.5):
	clf = MultinomialNB(alpha=_alpha)
	clf.fit(X_train, y_train)

	predictions_count = clf.predict(X_test)

	print("f1 score: ", f1_score(y_test, predictions_count))
	print("accuracy score: ", accuracy_score(y_test, predictions_count))

def pre_preprocessing_opt(df):
	# remove all html tags
	df_train['Comment'] = df_train['Comment'].str.replace(r'<.*?>', ' ')

	df_train = correct_comments(df_train)

	# remove one or two letters words
	df_train['Comment'] = df_train['Comment'].str.replace(r"\b[a-zA-Z]\b|\b[a-zA-Z][a-zA-Z]\b", "")

	# NO IMPROVEMENT, SO IT IS NOT USED
	# non english words include usernames which are used a lot in the comments
	# df_train = remove_non_english_comments(df_train)

	df['Comment'] = lemmatization(df['Comment'].tolist())

if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_cleaned.csv", sep=',')
	df_test = pd.read_csv("../data/test_cleaned.csv", sep=',')

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']
	
	######### BEAT THE BENCHMARK #########
	df_train = pre_preprocessing_opt(df_train)
	df_test = pre_preprocessing_opt(df_test)

	count_vectorizer = CountVectorizer(min_df=3)
	X_train = count_vectorizer.fit_transform(df_train['Comment'])
	X_test = count_vectorizer.transform(df_test['Comment'])

	run_naive_bayes(X_train.toarray(), y_train,
			X_test.toarray(), y_test)

# vectorizer: increase min_df and try vectorizer parameters