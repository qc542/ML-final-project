import pandas as pd
from sklearn import linear_model
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

# Shape of "features": (38821, 9)
# Shape of "target": (38821,)

# Allocating 70% of the dataset for training and the rest for testing
features_train, features_test, target_train, target_test = \
        model_selection.train_test_split(features, target, \
        test_size=0.3, random_state=32)

# The beginning of Logistic Regression with L1 Regularization
training_accuracy_lst = []
test_accuracy_lst = []
param_lst = []

def logistic_regression_l1(param, features_train, target_train, features_test, \
        target_test):
    model = linear_model.LogisticRegression(penalty='l1', C=param, \
            solver='saga')
    model.fit(features_train, target_train)
    predict_train = model.predict(features_train)
    training_accuracy = model.score(features_train, target_train)
    training_accuracy_lst.append(training_accuracy)
    print("Logistic regression accuracy on training data: %f" \
            % training_accuracy)

    predict_test = model.predict(features_test)
    test_accuracy = model.score(features_test, target_test)
    test_accuracy_lst.append(test_accuracy)
    print("Logistic regression accuracy on test data: %f" \
            % test_accuracy)
    param_lst.append(param)

param_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for p in param_values:
    logistic_regression_l1(p, features_train, target_train, features_test, \
            target_test)

plt.scatter(param_lst, training_accuracy_lst)
plt.plot(param_lst, training_accuracy_lst)
plt.xlabel('c')
plt.ylabel('Logistic Regression Accuracy on Training/Test Data')
plt.scatter(param_lst, test_accuracy_lst)
plt.plot(param_lst, test_accuracy_lst)
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.show()

# The beginning of Logistic Regression with L2 Regularization
training_accuracy_l2 = []
test_accuracy_l2 = []
param_lst_l2 = []

def logistic_regression_l2(param, features_train, target_train, \
        features_test, target_test):
    model = linear_model.LogisticRegression(C=param, solver='lbfgs')
    model.fit(features_train, target_train)

    predict_train = model.predict(features_train)
    training_accuracy = model.score(features_train, target_train)
    training_accuracy_l2.append(training_accuracy)
    print("Logistic regression accuracy on training data: %f" \
            % training_accuracy)

    predict_test = model.predict(features_test)
    test_accuracy = model.score(features_test, target_test)
    test_accuracy_l2.append(test_accuracy)
    print("Logistic regression accuracy on test data: %f" \
            % test_accuracy)
    param_lst_l2.append(param)

for p in param_values:
    logistic_regression_l2(p, features_train, target_train, \
            features_test, target_test)

plt.scatter(param_lst_l2, training_accuracy_l2)
plt.plot(param_lst_l2, training_accuracy_l2)
plt.xlabel('c')
plt.ylabel('Logistic Regression Accuracy on Training/Test Data')
plt.scatter(param_lst_l2, test_accuracy_l2)
plt.plot(param_lst_l2, test_accuracy_l2)
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper right')
plt.show()
