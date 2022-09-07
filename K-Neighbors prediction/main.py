###############################################################
# Prediction based on train data using K-Neighbors Classifier #
###############################################################

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

X = np.load('X_public400.npy', allow_pickle=True)
Xev = np.load('X_eval400.npy', allow_pickle=True)
y_public = np.load('y_public400.npy', allow_pickle=True)

#use LabelEncoder for string data
enc = LabelEncoder()
for i in range(180, 200):
    label_encoder = enc.fit(X[:, i])

for i in range(180, 200):
    X[:, i] = label_encoder.transform(X[:, i])
    Xev[:, i] = label_encoder.transform(Xev[:, i])


#replace all nan values by the average of each column
for i in range(len(X[0])):
    a = np.array(X[:,i])
    b = np.array(Xev[:, i])
    mean = np.mean(a[~np.isnan(a.astype(np.float64))])
    mean = np.mean(b[~np.isnan(b.astype(np.float64))])
    a[np.isnan(a.astype(np.float64))] = mean
    b[np.isnan(b.astype(np.float64))] = mean
    X[:,i] = a
    Xev[:,i] = b


Xn_p = np.zeros(shape = (600,180))
Xn_e = np.zeros(shape = (200,180))
for i in range(0,180):
    Xn_p[:,i] = X[:,i]
    Xn_e[:, i] = Xev[:, i]

j = 180
Xs_p = np.zeros(shape = (600,20))
Xs_e = np.zeros(shape = (200,20))
for i in range(20):
    Xs_p[:,i] = X[:,j]
    Xs_e[:,i] = Xev[:,j]
    j = j + 1


#use OneHotEncoder for string data
ohe = OneHotEncoder(sparse=False,handle_unknown='ignore').fit(Xs_p)
Xs_p = ohe.transform(Xs_p)
Xs_e = ohe.transform(Xs_e)


#useStandartScaler na to normalize data
scaler = StandardScaler()
scaled_data = scaler.fit(Xn_p)
Xp_p = scaler.transform(Xn_p)
Xp_e = scaler.transform(Xn_e)


pca = PCA()
Xp_p = pca.fit_transform(Xp_p)
Xp_e = pca.transform(Xp_e)

X_public = np.hstack((Xp_p,Xs_p))
X_eval = np.hstack((Xp_e,Xs_e))


#to validate the classifier I used train_test_split on datax X_public and y_public
X_train,X_test,y_train,y_test=train_test_split(X_public, y_public, test_size=0.1, random_state=50)

#use the KNN classifier itself together with GridSearch
knn = KNeighborsClassifier()

k_range = list(range(1,50))
weight_options = ['uniform', 'distance']
param_grid = dict(n_neighbors=k_range,weights = weight_options)
grid = GridSearchCV(knn, param_grid, cv=50, scoring='accuracy')

grid.fit(X_train, y_train)
y_predict = grid.predict(X_test)

print("Accuracy score: ",accuracy_score(y_test, y_predict)*100)

# Accuracy score:  91.66666666666666 #




