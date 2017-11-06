#单一决策树、随机森林分类、梯度上升决策树
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')


X=titanic[['pclass','age','sex']]
y=titanic['survived']

#补充age
X['age'].fillna(X['age'].mean(),inplace=True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

#单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict=dtc.predict(X_test)

#随机森林分类器
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)

#梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict=gbc.predict(X_test)

from sklearn.metrics import classification_report
print('The accuracy of DecisionTree Classifier is',dtc.score(X_test,y_test))
print(classification_report(y_test,dtc_y_predict,target_names=['died','survived']))

print('The accuracy of RandomForest Classifier is',rfc.score(X_test,y_test))
print(classification_report(y_test,rfc_y_predict,target_names=['died','survived']))

print('The accuracy of GradientBoosting Classifier is',gbc.score(X_test,y_test))
print(classification_report(y_test,gbc_y_predict,target_names=['died','survived']))
