from sklearn.datasets import load_digits
#通过数据加载器获得手写体数字数码图像
digits = load_digits()
print(digits.data.shape)

#分割训练集测试集
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    digits.data,digits.target,test_size=0.25,random_state=33)

print(y_train.shape)
print(y_test.shape)

#使用支持向量机
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc=LinearSVC()

lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

from sklearn.metrics import classification_report
print('Accuracy of Linear SVC:',lsvc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))