import pandas as pd
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("AB_NYC_2019.csv")
dataset = dataset.dropna()

# Convert features written as strings to numbers
for i in dataset.index:
    if dataset['room_type'][i] == "Entire home/apt":
        dataset.at[i, 'room_type'] = 1
    if dataset['room_type'][i] == "Private room":
        dataset.at[i, 'room_type'] = 2
    if dataset['room_type'][i] == "Shared room":
        dataset.at[i, 'room_type'] = 3
    if dataset['neighbourhood_group'][i] == "Manhattan":
        dataset.at[i, 'neighbourhood_group'] = 1
    if dataset['neighbourhood_group'][i] == "Brooklyn":
        dataset.at[i, 'neighbourhood_group'] = 2
    if dataset['neighbourhood_group'][i] == "Bronx":
        dataset.at[i, 'neighbourhood_group'] = 3
    if dataset['neighbourhood_group'][i] == "Staten Island":
        dataset.at[i, 'neighbourhood_group'] = 4
    if dataset['neighbourhood_group'][i] == "Queens":
        dataset.at[i, 'neighbourhood_group'] = 5

# Excluding irrelevant features from the dataset
features = dataset[['latitude', 'longitude', 'minimum_nights',\
        'number_of_reviews', 'reviews_per_month', \
        'calculated_host_listings_count', 'availability_365', \
        'room_type', 'neighbourhood_group']]
target = dataset[['price']]

features = features.sample(frac=0.03)
target = target.sample(frac=0.03)

# Convert prices to integer labels of price ranges
for i in target.index:
    if target['price'][i] < 200: target.at[i, 'price'] = 0
    elif (200 < target['price'][i]) and (target['price'][i] < 500):
        target.at[i, 'price'] = 1
    elif (500 < target['price'][i]) and (target['price'][i] < 1000):
        target.at[i, 'price'] = 2
    elif (1000 < target['price'][i]) and (target['price'][i] < 1500):
        target.at[i, 'price'] = 3 
    elif (1500 < target['price'][i]) and (target['price'][i] < 2000):
        target.at[i, 'price'] = 4
    elif (2000 < target['price'][i]) and (target['price'][i] < 3000):
        target.at[i, 'price'] = 5
    elif (3000 < target['price'][i]) and (target['price'][i] < 4000):
        target.at[i, 'price'] = 6
    else:
        target.at[i, 'price'] = 7

target = target.to_numpy()
target = target.reshape(target.shape[0])


# Shape of "features": (388, 9)
# Shape of "target": (388,)

# Allocating 70% of the dataset for training and the rest for testing
features_train, features_test, target_train, target_test = \
        model_selection.train_test_split(features, target, \
        test_size=0.3, random_state=32)

# The beginning of SVM with linear kernel
training_accuracy_lst = []
test_accuracy_lst = []
param_lst = []
weights = []

def support_vector_linear(param, weights):
    model_linear = svm.SVC(probability=False, kernel='linear', C=param)
    model_linear.fit(features_train, target_train)
    predict_train = model_linear.predict(features_train)
    training_accuracy = model_linear.score(features_train, target_train)
    training_accuracy_lst.append(training_accuracy)
    print('SVM (linear kernel) accuracy on training Data: \
            {0:f}'.format(training_accuracy))

    predict_test = model_linear.predict(features_test)
    test_accuracy = model_linear.score(features_test, target_test)
    test_accuracy_lst.append(test_accuracy)
    print('SVM (linear kernel) accuracy on test data: \
            {0:f}'.format(test_accuracy))
    param_lst.append(param)
    weights.append(model_linear.coef_)


param_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for p in param_values:
    support_vector_linear(p, weights)

plt.scatter(param_lst, training_accuracy_lst)
plt.plot(param_lst, training_accuracy_lst)
plt.xlabel('c')
plt.ylabel('SVM (linear) Accuracy on Training/Test Data')
plt.scatter(param_lst, test_accuracy_lst)
plt.plot(param_lst, test_accuracy_lst)
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper right')
plt.show()

""" This function calculates the L2 norm of each of the weight vectors and returns their average."""
def l2_norm_weights(weights):
    norms = []
    for n in weights:
        norm = np.sqrt(np.sum(n**2))
        norms.append(norm)
    return np.mean(norms)

l2_norm_lst = []
for weight in weights:
    l2_norm_lst.append(l2_norm_weights(weight))

plt.scatter(param_lst, l2_norm_lst)
plt.plot(param_lst, l2_norm_lst)
plt.xlabel('c')
plt.ylabel('Average L2 Norm of Feature Weights')
plt.show()

# The beginning of SVM with Radial Basis Function (RBF) kernel
training_accuracy_lst = []
test_accuracy_lst = []
param_lst = []
weights = []

def support_vector_rbf(param, weights):
    model_rbf = svm.SVC(probability=False, kernel='rbf', C=param)
    model_rbf.fit(features_train, target_train)
    predict_train = model_rbf.predict(features_train)
    training_accuracy = model_rbf.score(features_train, target_train)
    training_accuracy_lst.append(training_accuracy)
    print('SVM (RBF kernel) accuracy on training Data: \
            {0:f}'.format(training_accuracy))

    predict_test = model_rbf.predict(features_test)
    test_accuracy = model_rbf.score(features_test, target_test)
    test_accuracy_lst.append(test_accuracy)
    print('SVM (RBF kernel) accuracy on test data: \
            {0:f}'.format(test_accuracy))
    param_lst.append(param)
    weights.append(model_rbf.dual_coef_)


for p in param_values:
    support_vector_rbf(p, weights)

plt.scatter(param_lst, training_accuracy_lst)
plt.plot(param_lst, training_accuracy_lst)
plt.xlabel('c')
plt.ylabel('SVM (RBF) Accuracy on Training/Test Data')
plt.scatter(param_lst, test_accuracy_lst)
plt.plot(param_lst, test_accuracy_lst)
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper right')
plt.show()

l2_norm_lst = []
for weight in weights:
    l2_norm_lst.append(l2_norm_weights(weight))

plt.scatter(param_lst, l2_norm_lst)
plt.plot(param_lst, l2_norm_lst)
plt.xlabel('c')
plt.ylabel('Average L2 Norm of Feature Weights')
plt.show()


# The beginning of SVM with Polynomial kernel
training_accuracy_lst = []
test_accuracy_lst = []
param_lst = []
weights = []

def support_vector_poly(param, weights):
    model_poly = svm.SVC(probability=False, kernel='poly', C=param)
    model_poly.fit(features_train, target_train)
    predict_train = model_poly.predict(features_train)
    training_accuracy = model_poly.score(features_train, target_train)
    training_accuracy_lst.append(training_accuracy)
    print('SVM (Polynomial kernel) accuracy on training Data: \
            {0:f}'.format(training_accuracy))

    predict_test = model_poly.predict(features_test)
    test_accuracy = model_poly.score(features_test, target_test)
    test_accuracy_lst.append(test_accuracy)
    print('SVM (Polynomial kernel) accuracy on test data: \
            {0:f}'.format(test_accuracy))
    param_lst.append(param)
    weights.append(model_poly.dual_coef_)

for p in param_values:
    support_vector_poly(p, weights)

plt.scatter(param_lst, training_accuracy_lst)
plt.plot(param_lst, training_accuracy_lst)
plt.xlabel('c')
plt.ylabel('SVM (Polynomial) Accuracy on Training/Test Data')
plt.scatter(param_lst, test_accuracy_lst)
plt.plot(param_lst, test_accuracy_lst)
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper right')
plt.show()

l2_norm_lst = []
for weight in weights:
    l2_norm_lst.append(l2_norm_weights(weight))

plt.scatter(param_lst, l2_norm_lst)
plt.plot(param_lst, l2_norm_lst)
plt.xlabel('c')
plt.ylabel('Average L2 Norm of Feature Weights')
plt.show()
