
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
y_train = data['Survived']
data = data.append(pd.read_csv('test.csv'))
data = data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin', 'Survived'], axis = 1)
data['FamMem'] = data['SibSp'] + data['Parch']
#del data['SibSp']
#del data['Parch']

check = data.describe()
check_again = data.describe(include = 'all')
data['Embarked'] = data['Embarked'].fillna('S')
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Age'] = data['Age'].fillna(data['Age'].mean())
'''col = list(check_again)
for i in col :
    data[i] = data[i].fillna(data[i].value_counts().idxmax())'''
        
categorical = data.select_dtypes(exclude = ['number'])
columns_categorical = list(categorical)
        
col_ind = []
for i in columns_categorical:
    col_ind.append(data.columns.get_loc(i))
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in columns_categorical:
    labenc = LabelEncoder()
    data[i] = labenc.fit_transform(data[i])
ohenc = OneHotEncoder(categorical_features = col_ind)
data = ohenc.fit_transform(data).toarray()

X_train = data[:-418, :]
X_test = data[891:, :]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 118, gamma = 0.013, random_state = 0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

'''from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
accuracies.mean()
accuracies.std()'''

from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [118, 117, 116, 115], 'kernel' : ['poly', 'sigmoid', 'rbf'], 'gamma' : [0.01, 0.012, 0.013, 0.014, 0.02], 'random_state' : [0, 11, 42]}]
parameters1 = [{'C' : [118, 119, 120, 121, 122], 'kernel' : ['linear'], 'gamma' : [0.01, 0.012, 0.013, 0.014, 0.02],
               'random_state' : [0, 11, 42]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 3,
                           verbose = 60, n_jobs = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score1 = grid_search.best_score_
best_params = grid_search.best_params_

y_pred_test = grid_search.predict(X_test)