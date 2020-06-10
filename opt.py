import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score


def run_naive_bayes(X_train, y_train, X_test, y_test, _alpha=0.5):
	# clf = MultinomialNB(alpha=_alpha)
	clf = GaussianNB()
	clf.fit(X_train, y_train)

	predictions_count = clf.predict(X_test)
	print("f1 score: ", f1_score(y_test, predictions_count))
	print("accuracy score: ", accuracy_score(y_test, predictions_count))


if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_opt.csv", sep=',')
	df_train['Comment'] = df_train['Comment'].fillna(' ')
	df_test = pd.read_csv("../data/test_opt.csv", sep=',')
	df_test['Comment'] = df_test['Comment'].fillna(' ')

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']
	
	count_vectorizer = CountVectorizer(min_df=3)
	X_train = count_vectorizer.fit_transform(df_train['Comment'])
	X_test = count_vectorizer.transform(df_test['Comment'])

	run_naive_bayes(X_train.toarray(), y_train,X_test.toarray(), y_test, 1)