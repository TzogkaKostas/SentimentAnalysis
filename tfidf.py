import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("../data/train_cleaned.csv", sep=',')
	df_test = pd.read_csv("../data/test_cleaned.csv", sep=',')

	# labels
	y_train = df_train['Insult']
	y_test = df_test['Insult']

	# vectorization by TfidfVectorizer
	tfidf_vectorizer = TfidfVectorizer()
	X_train = tfidf_vectorizer.fit_transform(df_train['Comment'])
	X_test = tfidf_vectorizer.transform(df_test['Comment'])


	#################################### SVM ####################################
	clf = GridSearchCV(svm.SVC(), {'kernel':('linear', 'rbf'), 'C':[1, 10]}, n_jobs=-1)
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)

	print("f1 score: ", f1_score(y_test, predictions))
	print("accuracy score: ", accuracy_score(y_test, predictions))

	#################################### Random Forrest ####################################
	clf = RandomForestClassifier()
	clf.fit(X_train, y_train)

	predictions = clf.predict(X_test)

	print("f1 score: ", f1_score(y_test, predictions))
	print("accuracy score: ", accuracy_score(y_test, predictions))