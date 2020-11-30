import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Use StandardScaler to make the mean zero and the variance 1
scaler = StandardScaler()
features = scaler.fit_transform(features)

# allocating 70% of the dataset for training and the rest for testing
features_train, features_test, target_train, target_test = \
        model_selection.train_test_split(features, target, \
        test_size=0.3, random_state=32)

def target_vector_conversion(target):
    target_vector = np.zeros((target.shape[0], 8))
    # The target is an integer between 0 and 7, so there should be 8 neurons
    # in the output layer. If the predicted target is 1, the output layer
    # will be (0,1,0,0,0,0,0,0); so on and so forth.
    for i in range(len(target)):
        target_vector[i, target[i]] = 1
    return target_vector

target_vector_train = target_vector_conversion(target_train)
target_vector_test = target_vector_conversion(target_test)

