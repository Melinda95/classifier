import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#print(titanic.head())

#print(titanic.info())

X=titanic[['pclass','age','sex']]
y=titanic['survived']

print(X.info())

#补充age
X['age'].fillna(X['age'].mean(),inplace=True)
print(X.info())

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict=dtc.predict(X_test)

from sklearn.metrics import classification_report
print('The accuracy of DecisionTree Classifier is',dtc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=['died','survived']))
