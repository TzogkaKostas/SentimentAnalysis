import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from spellchecker import SpellChecker

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
		# if comment == '':

		# tokenize the comment and find POS tag for each word
		comment_with_tags = pos_tag(word_tokenize(comment))
		new_docs.append(_lemmatization(comment_with_tags))

	return new_docs

def pre_preprocessing_opt(df):
	# remove leading and trailing quotes (") 
	df['Comment'] = df['Comment'].str.strip('"')

	# convert all characters to lowercase
	df['Comment'] = df['Comment'].str.lower()

	# remove all URLs
	df['Comment'] = df['Comment'].str.replace(r'https?:\/\/.*[\r\n]*', ' ')

	# remove all html tags
	df['Comment'] = df['Comment'].str.replace(r'<.*?>', ' ')

	# remove usernames (@user123)	
	df['Comment'] = df['Comment'].str.replace(r'@([a-zA-Z0-9_])*', ' ')

	df = remove_escaping_characters(df)

	# remove all specials characters ('.', ',', '\n' ...)
	df['Comment'] = df['Comment'].str.replace('[^a-z]+', ' ')

	df = correct_comments(df)

	df['Comment'] = lemmatization(df['Comment'].tolist())

	# remove one or two letters words
	df['Comment'] = df['Comment'].str.replace(r"\b[a-zA-Z]\b|\b[a-zA-Z][a-zA-Z]\b", "")

	# NO IMPROVEMENT, SO IT IS NOT USED
	# non english words include usernames which are used a lot in the comments
	# df_train = remove_non_english_comments(df_train)

	return df

def remove_escaping_characters(df):
	# e.g. 'a\\xc2\\xa0majority of canadians can' is converted to
	# 'a majority of canadians can'
	new_rows = []
	for row in df['Comment']:
		# double decoding is needed in order to convert characters with double
		# slashes (\\) 

		# e.g. 'a\\xc2\\xa0majority of canadians can' is converted to
		# 'a\xc2\xa0majority of canadians can'
		new_row = row.encode().decode('unicode_escape')
		# and then is converted to 'a majority of canadians can'
		new_row = new_row.encode().decode('unicode_escape')

		new_rows.append(new_row)
	df['Comment'] = new_rows

	return df

def correct_comments(df):
	spell_checker = SpellChecker() 

	corrected_comments = []
	for comment in df['Comment']:
		comment = correct_comment(spell_checker, comment)
		corrected_comments.append(comment)
	
	df['Comment'] = corrected_comments

	return df

def correct_comment(spell_checker, comment):
	checked_comment = ""
	for word in word_tokenize(comment):
		checked_comment += spell_checker.correction(word) + " "

	return checked_comment

def remove_non_english_comments(df):
	try:
		english_words = set(corpus.words.words())
	except LookupError:
		download('words')

	english_comments= []
	for comment in df['Comment']:
		comment = " ".join(w for w in wordpunct_tokenize(comment) if w in english_words)

		english_comments.append(comment)

	df['Comment'] = english_comments

	return df


if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_cleaned.csv", sep=',')
	df_test = pd.read_csv("../data/test_cleaned.csv", sep=',')

	df_train = pre_preprocessing_opt(df_train)
	df_test = pre_preprocessing_opt(df_test)

	# create optimized files
	with open("../data/train_opt.csv", 'w+') as file:
		file.write(df_train.to_csv(index=False, sep=','))
	
	with open("../data/test_opt.csv", 'w+') as file:
		file.write(df_test.to_csv(index=False, sep=','))