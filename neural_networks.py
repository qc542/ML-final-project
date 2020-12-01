import pandas as pd
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import copy

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

# Use StandardScaler to make the mean zero and the variance 1
scaler = StandardScaler()
original_features = copy.deepcopy(features)
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

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def initialize_weights(network_structure):
    weights = {}
    biases = {}
    for layer in range(1, len(network_structure)):
        weights[layer] = rand.random_sample((network_structure[layer], \
                network_structure[layer-1]))
        biases[layer] = rand.random_sample((network_structure[layer],))
    return weights, biases

def initialize_gradients(network_structure):
    gradient_w = {}
    gradient_b = {}
    for layer in range(1, len(network_structure)):
        gradient_w[layer] = np.zeros((network_structure[layer], \
                network_structure[layer-1]))
        gradient_b[layer] = np.zeros((network_structure[layer],))
    return gradient_w, gradient_b

def feed_forward(n, weights, biases):
    a_values = {1:n}
    z_values = {}
    for layer in range(1, len(weights)+1):
        input_values = a_values[layer]
        z_values[layer+1] = weights[layer].dot(input_values) + \
                biases[layer]
        a_values[layer+1] = sigmoid(z_values[layer+1])
    return a_values, z_values

def output_layer_error(target, output_a, output_z):
    return -(target-output_a) * sigmoid_derivative(output_z)

def hidden_layer_error(next_error, current_weights, current_z):
    return np.dot(np.transpose(current_weights), next_error) * \
            sigmoid_derivative(current_z)

# Calculates the average L2 norm of a weight vector in the network
def l2_norm_weights(network_structure, weights):
    norms = []
    for layer in range(1, len(network_structure)):
        current_layer = weights[layer]
        for vec in current_layer:
            norm = np.sqrt(np.sum(vec**2))
            norms.append(norm)
    return np.mean(norms)

l2_norm_lst = []

def back_propagation(network_structure, features, target, \
        iterations=3000, alpha=0.25):
    weights, biases = initialize_weights(network_structure)
    current_iteration = 0
    num_examples = len(target)
    average_cost_lst = []
    print('Beginning gradient descent. Total number of iterations: {}'.format(iterations))
    while current_iteration < iterations:
        if current_iteration % 1000 == 0:
            print('Current iteration: No. {} of {}'.format(\
                    current_iteration, iterations))
        gradient_w, gradient_b = initialize_gradients(network_structure)
        average_cost = 0
        for i in range(num_examples):
            error = {}
            a_values, z_values = feed_forward(features[i,:], \
                    weights, biases)
            for layer in range(len(network_structure), 0, -1):
                if layer == len(network_structure):
                    error[layer] = output_layer_error(target[i,:], \
                            a_values[layer], z_values[layer])
                    average_cost += np.linalg.norm\
                            ((target[i,:]-a_values[layer]))
                else:
                    if layer > 1:
                        error[layer] = hidden_layer_error(error[layer+1],\
                                weights[layer], z_values[layer])
                    gradient_w[layer] += np.dot\
                            (error[layer+1][:,np.newaxis],\
                            np.transpose(a_values[layer][:,np.newaxis]))
                    gradient_b[layer] += error[layer+1]
        for layer in range(len(network_structure)-1, 0, -1):
            weights[layer] += -(alpha)*(1.0/num_examples*gradient_w[layer])
            biases[layer] += -(alpha)*(1.0/num_examples*gradient_b[layer])
        average_cost = 1.0/num_examples*average_cost
        average_cost_lst.append(average_cost)
        l2_norm_lst.append(l2_norm_weights(network_structure, weights))
        current_iteration += 1
    return weights, biases, average_cost_lst

def predict_target(weights, biases, features, layers_total):
    num_examples = features.shape[0]
    target = np.zeros((num_examples,))
    for i in range(num_examples):
        a_values, z_values = feed_forward(features[i,:], weights, biases)
        target[i] = np.argmax(a_values[layers_total])
    return target


# Begin running the neural network
network_structure = [9,40,8]
weights, biases, average_cost_lst = back_propagation(network_structure, \
        features_train, target_vector_train)

plt.plot(average_cost_lst)
plt.ylabel('Average Cost')
plt.xlabel('Back Propagation Iteration Number')
plt.show()

plt.plot(l2_norm_lst)
plt.ylabel('Average L2 Norm of Feature Weights')
plt.xlabel('Back Propagation Iteration Number')
plt.show()

target_predictions = predict_target(weights, biases, features_test, 3)
print('Prediction accuracy is {}%'.format(accuracy_score(target_test, \
        target_predictions)*100))

# Begin neural network with L1 regularization

# Calculates the sum of the absolute values of all terms in a neural 
# network's weight vector
def weights_abs_sum(network_structure, weights):
    abs_sum = 0
    for layer in range(1, len(network_structure)):
        current_layer = weights[layer]
        for i in current_layer:
            for j in i:
                abs_sum += abs(j)
    return abs_sum

l2_norm_lst = []

def back_propagation_l1(network_structure, features, target, lambda_val, \
        iterations=1000, alpha=0.25):
    weights, biases = initialize_weights(network_structure)
    current_iteration = 0
    num_examples = len(target)
    average_cost_lst = []
    print('Beginning gradient descent. Total number of iterations: {}'.format(iterations))
    while current_iteration < iterations:
        if current_iteration % 100 == 0:
            print('Current iteration: No. {} of {}'.format(\
                    current_iteration, iterations))
        gradient_w, gradient_b = initialize_gradients(network_structure)
        average_cost = 0
        for i in range(num_examples):
            error = {}
            a_values, z_values = feed_forward(features[i,:], \
                    weights, biases)
            for layer in range(len(network_structure), 0, -1):
                if layer == len(network_structure):
                    error[layer] = output_layer_error(target[i,:], \
                            a_values[layer], z_values[layer])
                    average_cost = average_cost + np.linalg.norm\
                            ((target[i,:]-a_values[layer])) + \
                            lambda_val*\
                            weights_abs_sum(network_structure, weights)
                else:
                    if layer > 1:
                        error[layer] = hidden_layer_error(error[layer+1],\
                                weights[layer], z_values[layer])
                    gradient_w[layer] += np.dot\
                            (error[layer+1][:,np.newaxis],\
                            np.transpose(a_values[layer][:,np.newaxis]))
                    gradient_b[layer] += error[layer+1]
        for layer in range(len(network_structure)-1, 0, -1):
            weights[layer] += -(alpha)*(1.0/num_examples*gradient_w[layer])
            biases[layer] += -(alpha)*(1.0/num_examples*gradient_b[layer])
        average_cost = 1.0/num_examples*average_cost
        average_cost_lst.append(average_cost)
        l2_norm_lst.append(l2_norm_weights(network_structure, weights))
        current_iteration += 1
    final_average_cost = average_cost_lst[-1]
    final_l2_norm = l2_norm_lst[-1]
    return weights, biases, final_average_cost, final_l2_norm


# Begin neural network with polynomial transformation
lambda_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
weights_lst = []
final_average_cost_lst = []
final_l2_norm_lst = []
accuracy_lst = []

polynomial = PolynomialFeatures(2)
scaler = StandardScaler()
features_train, features_test, target_train, target_test = \
        model_selection.train_test_split(original_features, target, \
        test_size=0.3, random_state=32)
features_train_t = polynomial.fit_transform(features_train)
features_test_t = polynomial.fit_transform(features_test)
features_train_t = scaler.fit_transform(features_train_t)
features_test_t = scaler.fit_transform(features_test_t)
network_structure = [features_train_t.shape[1], 40, 8]

for lambda_val in lambda_lst:
    weights, biases, final_average_cost, final_l2_norm = \
            back_propagation_l1(network_structure, features_train_t, \
            target_vector_train, lambda_val)
    weights_lst.append(weights)
    final_average_cost_lst.append(final_average_cost)
    final_l2_norm_lst.append(final_l2_norm)
    target_predictions = predict_target(weights, biases, features_test_t, 3)
    accuracy = accuracy_score(target_test, target_predictions)*100
    print('Prediction accuracy is {}%'.format(accuracy))
    accuracy_lst.append(accuracy)

plt.scatter(lambda_lst, accuracy_lst)
plt.plot(lambda_lst, accuracy_lst)
plt.ylabel('Accuracy (L1 Regularization)')
plt.xlabel('c')
plt.show()

plt.scatter(lambda_lst, final_l2_norm_lst)
plt.plot(lambda_lst, final_l2_norm_lst)
plt.ylabel('Average L2 Norm of Feature Weights')
plt.xlabel('c')
plt.show()

# Begin neural network with L2 Regularization
# Calculates the term to be multiplied by lambda
def l2_term(network_structure, weights):
    norms = []
    for layer in range(1, len(network_structure)):
        current_layer = weights[layer]
        for vec in current_layer:
            norm = np.sum(vec**2)
            norms.append(norm)
    return np.sum(norms)

def back_propagation_l2(network_structure, features, target, lambda_val, \
        iterations=1000, alpha=0.25):
    weights, biases = initialize_weights(network_structure)
    current_iteration = 0
    num_examples = len(target)
    average_cost_lst = []
    print('Beginning gradient descent. Total number of iterations: {}'.format(iterations))
    while current_iteration < iterations:
        if current_iteration % 100 == 0:
            print('Current iteration: No. {} of {}'.format(\
                    current_iteration, iterations))
        gradient_w, gradient_b = initialize_gradients(network_structure)
        average_cost = 0
        for i in range(num_examples):
            error = {}
            a_values, z_values = feed_forward(features[i,:], \
                    weights, biases)
            for layer in range(len(network_structure), 0, -1):
                if layer == len(network_structure):
                    error[layer] = output_layer_error(target[i,:], \
                            a_values[layer], z_values[layer])
                    average_cost = average_cost + np.linalg.norm\
                            ((target[i,:]-a_values[layer])) + \
                            lambda_val*\
                            l2_term(network_structure, weights)
                else:
                    if layer > 1:
                        error[layer] = hidden_layer_error(error[layer+1],\
                                weights[layer], z_values[layer])
                    gradient_w[layer] += np.dot\
                            (error[layer+1][:,np.newaxis],\
                            np.transpose(a_values[layer][:,np.newaxis]))
                    gradient_b[layer] += error[layer+1]
        for layer in range(len(network_structure)-1, 0, -1):
            weights[layer] += -(alpha)*(1.0/num_examples*gradient_w[layer])
            biases[layer] += -(alpha)*(1.0/num_examples*gradient_b[layer])
        average_cost = 1.0/num_examples*average_cost
        average_cost_lst.append(average_cost)
        l2_norm_lst.append(l2_norm_weights(network_structure, weights))
        current_iteration += 1
    final_average_cost = average_cost_lst[-1]
    final_l2_norm = l2_norm_lst[-1]
    return weights, biases, final_average_cost, final_l2_norm


# Begin running the neural network
lambda_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
weights_lst = []
final_average_cost_lst = []
final_l2_norm_lst = []
accuracy_lst = []


for lambda_val in lambda_lst:
    weights, biases, final_average_cost, final_l2_norm = \
            back_propagation_l2(network_structure, features_train_t, \
            target_vector_train, lambda_val)
    weights_lst.append(weights)
    final_average_cost_lst.append(final_average_cost)
    final_l2_norm_lst.append(final_l2_norm)
    target_predictions = predict_target(weights, biases, features_test_t, 3)
    accuracy = accuracy_score(target_test, target_predictions)*100
    print('Prediction accuracy is {}%'.format(accuracy))
    accuracy_lst.append(accuracy)

plt.scatter(lambda_lst, accuracy_lst)
plt.plot(lambda_lst, accuracy_lst)
plt.ylabel('Accuracy (L2 Regularization)')
plt.xlabel('c')
plt.show()

plt.scatter(lambda_lst, final_l2_norm_lst)
plt.plot(lambda_lst, final_l2_norm_lst)
plt.ylabel('Average L2 Norm of Feature Weights')
plt.xlabel('c')
plt.show()


# Begin neural network with transformation by MinMaxScaler
lambda_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
weights_lst = []
final_average_cost_lst = []
final_l2_norm_lst = []
accuracy_lst = []

mscaler = MinMaxScaler()
scaler = StandardScaler()
features_train_t = mscaler.fit_transform(features_train)
features_test_t = mscaler.fit_transform(features_test)
features_train_t = scaler.fit_transform(features_train_t)
features_test_t = scaler.fit_transform(features_test_t)
network_structure = [features_train_t.shape[1], 40, 8]

for lambda_val in lambda_lst:
    weights, biases, final_average_cost, final_l2_norm = \
            back_propagation_l1(network_structure, features_train_t, \
            target_vector_train, lambda_val)
    weights_lst.append(weights)
    final_average_cost_lst.append(final_average_cost)
    final_l2_norm_lst.append(final_l2_norm)
    target_predictions = predict_target(weights, biases, features_test_t, 3)
    accuracy = accuracy_score(target_test, target_predictions)*100
    print('Prediction accuracy is {}%'.format(accuracy))
    accuracy_lst.append(accuracy)

plt.scatter(lambda_lst, accuracy_lst)
plt.plot(lambda_lst, accuracy_lst)
plt.ylabel('Accuracy (L1 Regularization)')
plt.xlabel('c')
plt.show()

plt.scatter(lambda_lst, final_l2_norm_lst)
plt.plot(lambda_lst, final_l2_norm_lst)
plt.ylabel('Average L2 Norm of Feature Weights')
plt.xlabel('c')
plt.show()


lambda_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
weights_lst = []
final_average_cost_lst = []
final_l2_norm_lst = []
accuracy_lst = []


for lambda_val in lambda_lst:
    weights, biases, final_average_cost, final_l2_norm = \
            back_propagation_l2(network_structure, features_train_t, \
            target_vector_train, lambda_val)
    weights_lst.append(weights)
    final_average_cost_lst.append(final_average_cost)
    final_l2_norm_lst.append(final_l2_norm)
    target_predictions = predict_target(weights, biases, features_test_t, 3)
    accuracy = accuracy_score(target_test, target_predictions)*100
    print('Prediction accuracy is {}%'.format(accuracy))
    accuracy_lst.append(accuracy)

plt.scatter(lambda_lst, accuracy_lst)
plt.plot(lambda_lst, accuracy_lst)
plt.ylabel('Accuracy (L2 Regularization)')
plt.xlabel('c')
plt.show()

plt.scatter(lambda_lst, final_l2_norm_lst)
plt.plot(lambda_lst, final_l2_norm_lst)
plt.ylabel('Average L2 Norm of Feature Weights')
plt.xlabel('c')
plt.show()
