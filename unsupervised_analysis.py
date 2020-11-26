from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

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

dataset = dataset[['latitude', 'longitude', 'price', 'minimum_nights',\
        'number_of_reviews', 'reviews_per_month', \
        'calculated_host_listings_count', 'availability_365', \
        'room_type', 'neighbourhood_group']]
model = KMeans(n_clusters=5)
model.fit(dataset)

all_predictions = model.predict(dataset)
print(all_predictions.shape)
label = {0: 'blue', 1: 'green', 2: 'pink', 3:'red', 4:'gray'}
x_axis = dataset['longitude']
y_axis = dataset['latitude']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

y_axis = dataset['price']
x_axis = dataset['reviews_per_month']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Reviews per Month')
plt.ylabel('Price ($)')
plt.show()

y_axis = dataset['price']
x_axis = dataset['minimum_nights']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Minimum # of Nights')
plt.ylabel('Price ($)')
plt.show()

y_axis = dataset['price']
x_axis = dataset['neighbourhood_group']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price ($)')
plt.show()

y_axis = dataset['price']
x_axis = dataset['room_type']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Room Type')
plt.ylabel('Price ($)')
plt.show()

y_axis = dataset['room_type']
x_axis = dataset['minimum_nights']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Minimum # of Nights')
plt.ylabel('Room Type')
plt.show()

y_axis = dataset['number_of_reviews']
x_axis = dataset['minimum_nights']
fig, ax = plt.subplots()
scatter = ax.scatter(x_axis, y_axis, c=all_predictions)
legend = plt.legend(*scatter.legend_elements(), loc="lower left", \
        title="Clusters")
ax.add_artist(legend)
plt.xlabel('Minimum # of Nights')
plt.ylabel('Number of Reviews')
plt.show()
