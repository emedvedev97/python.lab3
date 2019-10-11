import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x_train = pd.read_csv("/home/emedvedev/pythonLab/lab2/train.csv")
x_test = pd.read_csv("/home/emedvedev/pythonLab/lab2/test.csv")
y_test = pd.read_csv("/home/emedvedev/pythonLab/lab2/gender_submission.csv")

y_test = y_test['Survived']
y_train = x_train['Survived']

x_test = pd.concat([y_test,  x_test ],  axis=1)
x_train.drop(['PassengerId'], axis=1, inplace=True)
x_test.drop(['PassengerId'], axis=1, inplace=True)


def masFilter(mas):
    '''
    mas[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    mas.drop(['Survived',  'Name', 'Ticket'], axis=1, inplace=True)
    mas.drop(['Cabin'], axis=1, inplace=True)
    mas['Embarked']
    mas['Age'].fillna(mas['Age'].median(), inplace=True)
    mas['Embarked'].fillna('S', inplace=True)
    mas = pd.concat([mas,  pd.get_dummies(mas['Embarked'], prefix="Embarked")],  axis=1)
    mas.drop(['Embarked'], axis=1, inplace=True)
    mas['Sex'] = pd.factorize(mas['Sex'])[0]
    '''
    mas = mas[["Sex","Age"]].copy()
    mas['Age'].fillna(mas['Age'].median(), inplace=True)
    #mas['Fare'].fillna(mas['Fare'].median(), inplace=True)
    mas['Sex'] = pd.factorize(mas['Sex'])[0]
    #print(mas)

    return mas

x_train = masFilter(x_train)
x_test = masFilter(x_test)

#x_test['Fare'].fillna(x_test['Fare'].median(), inplace=True)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(x_train, y_train)
print("Правильность на тренеровачном наборе: {:.2f}".format(grid.score (x_train, y_train)))
print("Наилучшие значения параметров: {}".format(grid.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(grid.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(grid.score(x_test, y_test)))

x_max = max(x_test['Sex']) + 1
x_min = min(x_test['Sex']) - 1
y_max = max(x_test['Age']) + 1
y_min = min(x_test['Age']) - 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

plt.subplot(1, 1, 1)
Z = grid.best_estimator_.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(x_test['Sex'], x_test['Age'], c=y_test, edgecolors="grey", s=50,cmap='coolwarm')
plt.show()

