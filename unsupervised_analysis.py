from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("AB_NYC_2019.csv")
dataset = dataset.dropna()
for i in dataset.index:
    if dataset['room_type'][i] == "Entire home/apt":
        dataset.at[i, 'room_type'] = 1
    elif dataset['room_type'][i] == "Private room":
        dataset.at[i, 'room_type'] = 2
    elif dataset['room_type'][i] == "Shared room":
        dataset.at[i, 'room_type'] = 3

dataset = dataset[['latitude', 'longitude', 'price', 'minimum_nights',\
        'number_of_reviews', 'reviews_per_month', \
        'calculated_host_listings_count', 'availability_365', \
        'room_type']]
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
