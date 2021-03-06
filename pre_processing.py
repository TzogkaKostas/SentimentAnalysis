import pandas as pd
from nltk import wordpunct_tokenize
from nltk import corpus
from nltk import download
import numpy as np


def pre_processing(df):
	# remove leading and trailing quotes (") 
	df['Comment'] = df['Comment'].str.strip('"')

	# convert all characters to lowercase
	df['Comment'] = df['Comment'].str.lower()

	# remove all URLs
	df['Comment'] = df['Comment'].str.replace(r'https?:\/\/.*[\r\n]*', ' ')

	df = remove_escaping_characters(df)

	# remove all specials characters ('.', ',', '\n' ...)
	df['Comment'] = df['Comment'].str.replace('[^a-z]+', ' ')

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


if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train.csv", sep=',')
	df_test = pd.read_csv("../data/impermium_verification_labels.csv", sep=',')

	# pre processing
	df_train = pre_processing(df_train)
	df_test = pre_processing(df_test)

	# create cleaned files
	with open("../data/train_cleaned.csv", 'w+') as file:
		file.write(df_train.to_csv(index=False, sep=','))
	
	with open("../data/test_cleaned.csv", 'w+') as file:
		file.write(df_test.to_csv(index=False, sep=','))