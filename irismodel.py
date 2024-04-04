from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



# fetch dataset 
from sklearn.datasets import load_iris
iris = load_iris()
  
features = iris['data']
targets = iris['target']   
# print(features)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new1 = np.array( [[ 4.4, 3.2, 1.3 ,0.2]])
X_new2 = np.array( [[ 6.3, 2.5, 5.0, 1.9]])

# We can use the predict function to use our model to offer a prediction as to
# what species our X_new corresponds to.
prediction = knn.predict(X_new1)

print("Prediction for {} : {}".format(X_new1, iris['target_names'][prediction]))

prediction2 = knn.predict(X_new2)

print("Prediction for {}: {}".format(X_new2, iris['target_names'][prediction2]))


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
