from sklearn.datasets import fetch_20newsgroups  #新闻数据抓取器
news = fetch_20newsgroups(subset='all') #即时从互联网下载数据

print(len(news.data))
print(news.data[0])

#分离训练集和测试集
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size = 0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer   #文本特征向量转化
vec = CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

#分类预测
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)

#性能评估
from sklearn.metrics import  classification_report
print('The accuracy of Naive Bayes Classifier is',mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))