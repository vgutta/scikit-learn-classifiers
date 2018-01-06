from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


DTClf = tree.DecisionTreeClassifier()

DTClf = DTClf.fit(X,Y)

DTCprediction = DTClf.predict([[190, 70 ,43]])

print("Decision Tree Classifier:")
print(DTCprediction)

NBclf = GaussianNB()

NBclf = NBclf.fit(X,Y)

NBprediction = NBclf.predict([[190, 70 ,43]])

print("GaussianNB classifier:")
print(NBprediction)

SVMclf = SVC()

SVMclf = SVMclf.fit(X,Y)

SVMprediction = SVMclf.predict([[190, 70 ,43]])

print("SVM classifier:")
print(SVMprediction)
